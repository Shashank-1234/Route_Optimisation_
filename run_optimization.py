import pandas as pd
import numpy as np
from pyvrp import Model
from pyvrp.stop import MaxIterations
import sys
import os

def load_data(filepath):
    """
    Load distance and time matrices from the Excel file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    print(f"Loading data from {filepath}...")
    distance_matrix = pd.read_excel(filepath, sheet_name='Distance_Matrix', index_col=0)
    time_matrix = pd.read_excel(filepath, sheet_name='Time_Matrix', index_col=0)
    
    return distance_matrix, time_matrix

def calculate_emissions_matrix(distance_matrix, time_matrix):
    """
    Calculate the emissions matrix based on distance and speed.
    Formula: E = CCF(v) * EF * D
    """
    print("Calculating emissions matrix...")
    # Convert distance from meters to miles
    distance_matrix_miles = distance_matrix * 0.000621371
    
    # Convert time from seconds to hours
    time_matrix_hours = time_matrix / 3600
    
    # Calculate speed in miles per hour (mph)
    # Handle division by zero
    speed_matrix = distance_matrix_miles / time_matrix_hours
    speed_matrix = speed_matrix.fillna(0)
    
    # Parameters for CO2 emissions (EMFAC CARB model)
    A, B = -0.025952, 0.000309
    baseline_emission_factor = 0.76239  # g/m for BS-II trucks
    v0 = 27.4 # reference speed
    
    # Function to calculate emissions
    def get_emission(distance, speed):
        if distance <= 0 or speed <= 0:
            return 0
        ccf = np.exp(A * (speed - v0) + B * (speed - v0)**2)
        return baseline_emission_factor * ccf * distance

    emission_matrix = pd.DataFrame(index=distance_matrix.index, columns=distance_matrix.columns)
    
    for i in distance_matrix.index:
        for j in distance_matrix.columns:
            dist = distance_matrix.loc[i, j]
            speed = speed_matrix.loc[i, j]
            emission_matrix.loc[i, j] = get_emission(dist, speed)
            
    return emission_matrix

def solve_vrp(distance_matrix, emission_matrix, num_vehicles=18, capacity=10):
    """
    Solve the CVRP using PyVRP.
    """
    print("Setting up VRP model...")
    m = Model()
    m.add_vehicle_type(num_available=num_vehicles, capacity=capacity)
    
    # Clients and Depot
    # We assume the first index (Point_0) is the depot and the rest are clients.
    # We don't have the exact demand distribution without the shapefiles, 
    # so we will assume a default demand of 1 for simplicity, or 
    # try to approximate the paper's distribution if possible.
    # Paper: Ward 26, 27 (demand 1), Ward 30 (demand 2).
    
    num_points = len(distance_matrix)
    depot_idx = 0 # Point_0
    
    # Create locations/clients
    # Note: PyVRP 0.6+ uses add_client(x, y, delivery=...)
    # Since we lack coordinates, we pass (0,0). The cost is defined by the matrix anyway.
    
    locations = []
    
    # Add depot
    depot = m.add_depot(x=0, y=0)
    locations.append(depot)
    
    # Add clients
    # TODO: Refine demands based on actual ward data if available.
    default_demands = [1] * (num_points - 1)
    
    for i in range(len(default_demands)):
        demand = default_demands[i]
        client = m.add_client(x=0, y=0, delivery=demand)
        locations.append(client)
        
    # Setup cost matrix (Distance/Emission)
    profile = m.add_profile()
    
    print("Populating cost matrix...")
    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                continue
            
            # Use emission as the cost
            cost = emission_matrix.iloc[i, j]
            # Must be integer for PyVRP? 
            # PyVRP supports float costs? 
            # In only_emissions_pyvrp.py: m.add_edge(..., distance=float(cost))
            # PyVRP typically expects integer distances if using standard precision, 
            # but newer versions might support float or we can scale it.
            # Let's check imports. from pyvrp import Model.
            # If it's the latest version, it might want integers.
            # We will scale by 100 to keep some precision if needed, or just use int.
            # The previous code used float. Let's try float.
            
            m.add_edge(locations[i], locations[j], distance=int(cost * 100), profile=profile)
            
    print("Solving...")
    res = m.solve(stop=MaxIterations(2000), seed=42)
    return res, distance_matrix, emission_matrix

def print_results(res, distance_matrix, emission_matrix):
    if not res.best:
        print("No feasible solution found.")
        return

    print("\n" + "="*40)
    print("OPTIMIZATION RESULTS")
    print("="*40)
    
    total_emissions = 0
    total_distance = 0
    total_duration = 0 # We need time matrix for this, assume we can get it or just skip
    
    # Re-read time matrix for reporting if needed, or pass it in. 
    # For now, just report emissions and distance.
    
    for i, route in enumerate(res.best.routes()):
        route_emissions = 0
        route_distance = 0
        
        # Route is a list of client indices (1-based from PyVRP perspective usually, but let's check)
        # PyVRP route objects are iterable of clients.
        
        # We need to map client objects back to indices in our matrix.
        # The locations list was [depot, client1, client2...]
        # client1 corresponds to Point_1 (index 1 in matrix)
        
        prev_idx = 0 # Depot (Point_0)
        route_seq = [0]
        
        for client_idx in route:
            # client.idx is internal index. 
            # if depot is 0, client 1 is 1.
            curr_idx = client_idx
            route_seq.append(curr_idx)
            
            dist = distance_matrix.iloc[prev_idx, curr_idx]
            emis = emission_matrix.iloc[prev_idx, curr_idx]
            
            route_distance += dist
            route_emissions += emis
            
            prev_idx = curr_idx
            
        # Return to depot
        dist = distance_matrix.iloc[prev_idx, 0]
        emis = emission_matrix.iloc[prev_idx, 0]
        route_distance += dist
        route_emissions += emis
        route_seq.append(0)
        
        total_emissions += route_emissions
        total_distance += route_distance
        
        print(f"Vehicle {i+1}:")
        print(f"  Route: {route_seq}")
        print(f"  Emissions: {route_emissions:.2f} g")
        print(f"  Distance:  {route_distance:.2f} m")
        
    print("-" * 40)
    print(f"TOTAL EMISSIONS: {total_emissions:.2f} g")
    print(f"TOTAL DISTANCE:  {total_distance:.2f} m")
    print("=" * 40)

def main():
    data_file = 'data/travel_data.xlsx'
    
    try:
        dist_mat, time_mat = load_data(data_file)
        emis_mat = calculate_emissions_matrix(dist_mat, time_mat)
        res, _, _ = solve_vrp(dist_mat, emis_mat)
        print_results(res, dist_mat, emis_mat)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
