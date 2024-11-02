'''
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
import geopandas as gpd
import pandas as pd
import numpy as np

# Function to calculate emissions
def calculate_emissions(distance, speed):
    A, B = -0.025952, 0.000309
    baseline_emission_factor = 0.76239  # g/m for BS-II trucks from the CPCB report
    CCF = np.exp(A * (speed - 27.4) + B * (speed - 27.4)**2)
    return baseline_emission_factor * CCF * distance

def create_data_model():
    # Load the distance and time matrices from the Excel file
    excel_file = 'Algo/gmaps_distance_time_matrix_pooled.xlsx'
    distance_matrix = pd.read_excel(excel_file, sheet_name='Distance_Matrix', index_col=0)
    time_matrix = pd.read_excel(excel_file, sheet_name='Time_Matrix', index_col=0)

    # Convert distance from meters to miles
    distance_matrix_miles = distance_matrix * 0.000621371

    # Convert time from seconds to hours
    time_matrix_hours = time_matrix / 3600

    # Calculate speed in miles per hour (mph)
    speed_matrix = distance_matrix_miles / time_matrix_hours

    # Initialize emission matrix (empty DataFrame same size as distance matrix)
    emission_matrix = pd.DataFrame(index=distance_matrix.index, columns=distance_matrix.columns)

    # Calculate emissions for each route segment
    for i in distance_matrix.index:
        for j in distance_matrix.columns:
            distance = distance_matrix.loc[i, j]
            speed = speed_matrix.loc[i, j]
            if distance > 0 and speed > 0:  # Avoid self-loops
                emission_matrix.loc[i, j] = calculate_emissions(distance, speed)
            else:
                emission_matrix.loc[i, j] = 0

    # Compactor (depot) location
    compactor_location = (23.799829187950845, 86.44036347325256)

    # Load shapefiles for Ward 26 and Ward 27 (with demand = 1)
    shapefiles = ['Algo/clean_26.shp', 'Algo/final ward 27 intersection point.shp']
    coordinates = []
    DEMANDS = []

    for shapefile in shapefiles:
        wards_shp = gpd.read_file(shapefile)
        wards_shp = wards_shp[wards_shp['geometry'].notnull()]
        wards_shp['coords'] = wards_shp['geometry'].apply(
            lambda geom: (geom.y, geom.x) if geom is not None else None
        )
        coordinates.extend(wards_shp['coords'].tolist())
        # Add demand of 1 for these coordinates (Ward 26 and Ward 27)
        DEMANDS.extend([1] * len(wards_shp))

    # Load Ward 30 shapefile with demand = 2
    ward_30_shp = gpd.read_file('Algo/Ward30_Finalpoints.shp')
    ward_30_shp = ward_30_shp[ward_30_shp['geometry'].notnull()]
    ward_30_shp['coords'] = ward_30_shp['geometry'].apply(
        lambda geom: (geom.y, geom.x) if geom is not None else None
    )
    coordinates.extend(ward_30_shp['coords'].tolist())
    # Add demand of 2 for Ward 30
    DEMANDS.extend([1] * len(ward_30_shp))

    # Add depot info
    coordinates.insert(0, compactor_location)
    DEMANDS.insert(0, 0)  # Depot has demand 0

    # Prepare the data dictionary
    data = {
        'emission_matrix': emission_matrix.values,  # Use emission matrix for optimization
        'distance_matrix': distance_matrix.values,  # Include distance matrix for printing
        'time_matrix': time_matrix.values,  # Include time matrix for printing
        'demands': DEMANDS,
        'vehicle_capacities': [10] * 18,  # Number of vehicles and capacities
        'num_vehicles': 18,
        'depot': 0  # Index of depot
    }

    return data

def print_solution(manager, routing, solution, data):
    """Prints solution on console."""
    total_emission = 0
    total_distance = 0
    total_duration = 0
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_emission = 0
        route_distance = 0
        route_duration = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            plan_output += ' {} -> '.format(node_index)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            
            # Calculate emissions
            route_emission += data['emission_matrix'][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            
            # Calculate distance
            route_distance += data['distance_matrix'][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
            
            # Calculate duration (time)
            route_duration += data['time_matrix'][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]
        
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {:.2f} meters\n'.format(route_distance)
        plan_output += 'Duration of the route: {:.2f} seconds\n'.format(route_duration)
        plan_output += 'Emission of the route: {:.2f} g\n'.format(route_emission)
        
        print(plan_output)
        
        # Add route results to totals
        total_emission += route_emission
        total_distance += route_distance
        total_duration += route_duration
    
    # Print total emissions, distances, and durations
    print('Total distance of all routes: {:.2f} meters'.format(total_distance))
    print('Total duration of all routes: {:.2f} seconds'.format(total_duration))
    print('Total emissions of all routes: {:.2f} g'.format(total_emission))


# Create the routing model
def solve_vrp():
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['emission_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Define a callback to calculate emissions for each route segment
    def emission_callback(from_index, to_index):
        """Returns the emission between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['emission_matrix'][from_node][to_node]

    emission_callback_index = routing.RegisterTransitCallback(emission_callback)

    # Set the cost of travel for all vehicles to the emission callback
    routing.SetArcCostEvaluatorOfAllVehicles(emission_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING
    )
    search_parameters.time_limit.FromSeconds(30)
    search_parameters.log_search = True
    

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(manager, routing, solution, data)  # Pass 'data' here

# Execute the VRP
solve_vrp()
'''

"""
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import pandas as pd
import numpy as np
import geopandas as gpd

# Load the distance and emission matrices from the Excel file
excel_file = 'Algo/gmaps_distance_time_matrix_pooled.xlsx'
distance_matrix = pd.read_excel(excel_file, sheet_name='Distance_Matrix', index_col=0)
time_matrix = pd.read_excel(excel_file, sheet_name='Time_Matrix', index_col=0)

# Convert distance to miles and time to hours
distance_matrix_miles = distance_matrix * 0.000621371
time_matrix_hours = time_matrix / 3600
speed_matrix = distance_matrix_miles / time_matrix_hours

# Define emissions calculation
A, B = -0.025952, 0.000309
baseline_emission_factor = 0.76239

def calculate_emissions(distance, speed):
    CCF = np.exp(A * (speed - 27.4) + B * (speed - 27.4)**2)
    return baseline_emission_factor * CCF * distance

# Create emission matrix
emission_matrix = pd.DataFrame(index=distance_matrix.index, columns=distance_matrix.columns)
for i in distance_matrix.index:
    for j in distance_matrix.columns:
        dist = distance_matrix.loc[i, j]
        speed = speed_matrix.loc[i, j]
        emission_matrix.loc[i, j] = calculate_emissions(dist, speed) if dist > 0 and speed > 0 else 0

# Define your data model for OR-Tools
def create_data_model():
    data = {}
    data['distance_matrix'] = emission_matrix.values.astype(float)
    data['demands'] = [0] + [1] * (len(data['distance_matrix']) - 1)  # Replace with actual demands
    data['vehicle_capacities'] = [10] * 18  # Example capacity, adjust as needed
    data['num_vehicles'] = 18
    data['depot'] = 0
    return data

# Solution printing function
def print_solution(data, manager, routing, solution):
    total_emissions = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_emissions = 0
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route.append(node)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(index):
                route_emissions += data['distance_matrix'][previous_index][index]
        total_emissions += route_emissions
        print(f"Vehicle {vehicle_id} route: {route}")
        print(f"Vehicle {vehicle_id} emissions: {route_emissions:.2f} g")
    print(f"Total emissions: {total_emissions:.2f} g")

# Main function to solve the problem
def main():
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Emission callback
    def emission_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(emission_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Demand callback
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Set up search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(30)  # Adjust the time limit as needed

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
    else:
        print("No solution found!")

if __name__ == "__main__":
    main()
"""

"""Simple Vehicles Routing Problem (VRP).

   This is a sample using the routing library python wrapper to solve a VRP
   problem.
   A description of the problem can be found here:
   http://en.wikipedia.org/wiki/Vehicle_routing_problem.

   Distances are in meters.
"""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def create_data_model():
    # Load the distance and time matrices from the Excel file
    excel_file = 'Algo/gmaps_distance_time_matrix_pooled.xlsx'
    distance_matrix = pd.read_excel(excel_file, sheet_name='Distance_Matrix', index_col=0)
    time_matrix = pd.read_excel(excel_file, sheet_name='Time_Matrix', index_col=0)

    # Convert distance from meters to miles
    distance_matrix_miles = distance_matrix * 0.000621371

    # Convert time from seconds to hours
    time_matrix_hours = time_matrix / 3600

    # Calculate speed in miles per hour (mph)
    speed_matrix = distance_matrix_miles / time_matrix_hours

    # Initialize emission matrix (empty DataFrame same size as distance matrix)
    emission_matrix = pd.DataFrame(index=distance_matrix.index, columns=distance_matrix.columns)

    # Calculate emissions for each route segment
    for i in distance_matrix.index:
        for j in distance_matrix.columns:
            distance = distance_matrix.loc[i, j]
            speed = speed_matrix.loc[i, j]
            if distance > 0 and speed > 0:  # Avoid self-loops
                emission_matrix.loc[i, j] = calculate_emissions(distance, speed)
            else:
                emission_matrix.loc[i, j] = 0

    # Compactor (depot) location
    compactor_location = (23.799829187950845, 86.44036347325256)

    # Load shapefiles for Ward 26 and Ward 27 (with demand = 1)
    shapefiles = ['Algo/clean_26.shp', 'Algo/final ward 27 intersection point.shp']
    coordinates = []
    DEMANDS = []

    for shapefile in shapefiles:
        wards_shp = gpd.read_file(shapefile)
        wards_shp = wards_shp[wards_shp['geometry'].notnull()]
        wards_shp['coords'] = wards_shp['geometry'].apply(
            lambda geom: (geom.y, geom.x) if geom is not None else None
        )
        coordinates.extend(wards_shp['coords'].tolist())
        # Add demand of 1 for these coordinates (Ward 26 and Ward 27)
        DEMANDS.extend([1] * len(wards_shp))

    # Load Ward 30 shapefile with demand = 2
    ward_30_shp = gpd.read_file('Algo/Ward30_Finalpoints.shp')
    ward_30_shp = ward_30_shp[ward_30_shp['geometry'].notnull()]
    ward_30_shp['coords'] = ward_30_shp['geometry'].apply(
        lambda geom: (geom.y, geom.x) if geom is not None else None
    )
    coordinates.extend(ward_30_shp['coords'].tolist())
    # Add demand of 2 for Ward 30
    DEMANDS.extend([1] * len(ward_30_shp))

    # Add depot info
    coordinates.insert(0, compactor_location)
    DEMANDS.insert(0, 0)  # Depot has demand 0

    # Prepare the data dictionary
    data = {}
    data["distance_matrix"] = [distance_matrix.values]
    data['capacity'] = 10
    data['demands'] = DEMANDS

    return data

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data["distance_matrix"] = [
        # fmt: off
      [0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662],
      [548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210],
      [776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754],
      [696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358],
      [582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244],
      [274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708],
      [502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480],
      [194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856],
      [308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514],
      [194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468],
      [536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354],
      [502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844],
      [388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730],
      [354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536],
      [468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194],
      [776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798],
      [662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0],
        # fmt: on
    ]
    data["num_vehicles"] = 4
    data["depot"] = 0
    return data


def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    print(f"Objective: {solution.ObjectiveValue()}")
    max_route_distance = 0
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += f" {manager.IndexToNode(index)} -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
        plan_output += f"{manager.IndexToNode(index)}\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        print(plan_output)
        max_route_distance = max(route_distance, max_route_distance)
    print(f"Maximum of the route distances: {max_route_distance}m")



def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["distance_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = "Distance"
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        3000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name,
    )
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        print_solution(data, manager, routing, solution)
    else:
        print("No solution found !")


if __name__ == "__main__":
    main()