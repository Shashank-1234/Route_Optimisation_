from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import pandas as pd

# Load distance matrix from Excel
file_path = "Algo/distance_matrix.xlsx"
distance_matrix_df = pd.read_excel(file_path)

# Convert to numpy array (without index column)
distance_matrix = distance_matrix_df.drop(columns=['Unnamed: 0']).values

# Convert the matrix to kilometers if needed
distance_matrix_km = distance_matrix / 1000

# Number of bins (including depot)
n = distance_matrix.shape[0]

# Service time per bin (in minutes)
service_time_per_bin = 10

# Truck parameters
num_trucks = 2
working_hours = 8 * 60  # 8 hours in minutes
speed_kmh = 25  # Speed in km/h

# Calculate travel time matrix (in minutes)
def calculate_travel_time_km(distance_km):
    return (distance_km / speed_kmh) * 60

travel_time_matrix = [[calculate_travel_time_km(dist) for dist in row] for row in distance_matrix_km]

# Set up Google OR-Tools VRP solver
def create_data_model():
    data = {}
    data['distance_matrix'] = travel_time_matrix  # Use travel time as the distance
    data['num_vehicles'] = num_trucks
    data['depot'] = 0
    return data

data = create_data_model()

# Create the routing index manager
manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])

# Create the routing model
routing = pywrapcp.RoutingModel(manager)

# Define the cost of each arc
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)

# Set the cost of travel between nodes
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Add a capacity constraint for time (max 8 hours of work, including travel and service)
def time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return data['distance_matrix'][from_node][to_node] + service_time_per_bin

time_callback_index = routing.RegisterTransitCallback(time_callback)

# Set up the time constraint
time = 'Time'
routing.AddDimension(
    time_callback_index,
    0,  # no slack
    working_hours,  # max travel time
    True,  # start cumul to zero
    time)

time_dimension = routing.GetDimensionOrDie(time)

# Define search parameters
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

# Solve the problem
solution = routing.SolveWithParameters(search_parameters)

# Print the solution
if solution:
    print('Solution found!')
    total_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            plan_output += ' {} ->'.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        plan_output += ' {}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {} minutes\n'.format(route_distance)
        print(plan_output)
        total_distance += route_distance
    print('Total distance of all routes: {} minutes'.format(total_distance))
else:
    print('No solution found!')
