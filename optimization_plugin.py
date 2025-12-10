import pandas as pd
import numpy as np
from pyvrp import Model
from pyvrp.stop import MaxIterations
import sys

# Parameters for CO2 emissions (EMFAC CARB model)
A, B = -0.025952, 0.000309
BASELINE_EF = 0.76239  # g/m
V0 = 27.4 # reference speed

def calculate_emissions_matrix(distance_matrix, time_matrix):
    """
    Python implementation of the emissions model.
    """
    # Convert lists to numpy arrays
    dist = np.array(distance_matrix)
    time = np.array(time_matrix)
    
    # Avoid division by zero
    time = np.where(time == 0, 1e-9, time)
    
    # Units: Meters -> Miles, Seconds -> Hours
    dist_miles = dist * 0.000621371
    time_hours = time / 3600.0
    
    speed_mph = dist_miles / time_hours
    
    # CCF Calculation
    speed_diff = speed_mph - V0
    ccf = np.exp(A * speed_diff + B * speed_diff**2)
    
    # E = CCF * EF * D
    # Ensure non-negative
    emissions = BASELINE_EF * ccf * dist
    emissions = np.maximum(emissions, 0)
    
    return emissions

def solve_mission(distance_matrix, time_matrix):
    """
    Main entry point called from C++.
    Args:
        distance_matrix: List of lists (from C++)
        time_matrix: List of lists (from C++)
    Returns:
        dict: containing 'total_emissions', 'routes' (list of lists)
    """
    print("[Python] Received mission data from C++ Host.")
    print(f"[Python] Matrix Size: {len(distance_matrix)}x{len(distance_matrix[0])}")
    
    emissions = calculate_emissions_matrix(distance_matrix, time_matrix)
    
    # Setup PyVRP
    m = Model()
    m.add_vehicle_type(num_available=18, capacity=10)
    
    # We assume uniform demand for simplicity as per previous logic
    # and 0,0 coordinates since we operate on the matrix directly
    depot = m.add_depot(x=0, y=0)
    
    num_clients = len(distance_matrix) - 1
    clients = [m.add_client(x=0, y=0, delivery=1) for _ in range(num_clients)]
    locations = [depot] + clients
    
    profile = m.add_profile()
    
    # Populate Matrix
    # We scale by 100 to preserve 2 decimal precision in integer solver
    scale_factor = 100
    
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            if i != j:
                cost = int(emissions[i][j] * scale_factor)
                m.add_edge(locations[i], locations[j], distance=cost, profile=profile)
                
    print("[Python] Solving VRP...")
    res = m.solve(stop=MaxIterations(2000), seed=42)
    
    if not res.best:
        return {"status": "failed"}
        
    # Extract results
    routes = []
    total_emissions = 0
    
    for route in res.best.routes():
        # route is iterable of indices (PyVRP < 0.9 returns objects, but iterable yields indices? 
        # Actually newer PyVRP route objects yield clients.
        # client.idx 0 is depot.
        # We want the client indices.
        
        # Let's handle the extraction carefully based on PyVRP version installed
        r_indices = []
        for client in route:
             # In newer PyVRP, iterating a route yields Client objects
             # In some versions, it yields integers. 
             # We try both.
             try:
                 idx = client.idx
             except AttributeError:
                 idx = client
             r_indices.append(int(idx)) # Ensure int
             
        routes.append(r_indices)
    
    print(f"[Python] Returning {len(routes)} routes.")
    # Debug print first route
    if routes:
        print(f"[Python] Route 1 type: {type(routes[0])}, Element type: {type(routes[0][0])}")

    # Recalculate exact float emissions for reporting
    # (PyVRP objective is integer scaled)
    final_objective = res.best.cost() / scale_factor
        
    return {
        "status": "success",
        "total_emissions": final_objective,
        "routes": routes
    }
