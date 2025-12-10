from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
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
    """Stores the data for the problem, using the emission matrix."""
    # Load the distance and time matrices from the Excel file
    excel_file = 'Algo/gmaps_distance_time_matrix_pooled.xlsx'
    distance_matrix = pd.read_excel(excel_file, sheet_name='Distance_Matrix', index_col=0)
    time_matrix = pd.read_excel(excel_file, sheet_name='Time_Matrix', index_col=0)

    # Convert distance from meters to miles
    distance_matrix_miles = distance_matrix * 0.000621371
    time_matrix_hours = time_matrix / 3600  # Convert time to hours
    speed_matrix = distance_matrix_miles / time_matrix_hours.replace([np.inf, -np.inf], 0).fillna(0)

    # Calculate the emission matrix
    emission_matrix = pd.DataFrame(0, index=distance_matrix.index, columns=distance_matrix.columns)
    for i in distance_matrix.index:
        for j in distance_matrix.columns:
            distance = distance_matrix.loc[i, j]
            speed = speed_matrix.loc[i, j]
            if distance > 0 and speed > 0:  # Avoid self-loops
                emission_matrix.loc[i, j] = calculate_emissions(distance, speed)

    # Load shapefiles and extract coordinates for Ward 26, 27, and 30
    shapefiles = ['Algo/clean_26.shp', 'Algo/final ward 27 intersection point.shp', 'Algo/Ward30_Finalpoints.shp']
    DEMANDS = []
    compactor_location = (23.799829187950845, 86.44036347325256)
    coordinates = [compactor_location]
    DEMANDS.append(0)  # Depot has demand 0

    for shapefile in shapefiles:
        wards_shp = gpd.read_file(shapefile)
        wards_shp = wards_shp[wards_shp['geometry'].notnull()]
        wards_shp['coords'] = wards_shp['geometry'].apply(lambda geom: (geom.y, geom.x))
        coordinates.extend(wards_shp['coords'].tolist())
        DEMANDS.extend([1] * len(wards_shp) if 'Ward30' not in shapefile else [2] * len(wards_shp))

    data = {
        'emission_matrix': emission_matrix.values,  # Use emission matrix as cost
        'demands': DEMANDS,
        'vehicle_capacities': [10] * 18,  # Define vehicle capacities
        'num_vehicles': 18,
        'depot': 0
    }
    return data

def print_solution(data, manager, routing, solution):
    """Prints solution on console and calculates total emissions."""
    total_emission = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f'Route for vehicle {vehicle_id}:\n'
        route_emission = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            plan_output += f' {node_index} -> '
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            next_node_index = manager.IndexToNode(index)
            route_emission += data['emission_matrix'][node_index][next_node_index]
        
        plan_output += f'{manager.IndexToNode(index)}\n'
        plan_output += f'Emission of the route: {route_emission:.2f} g\n'
        print(plan_output)
        total_emission += route_emission
    
    print(f'Total emissions of all routes: {total_emission:.2f} grams')

def main():
    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data['emission_matrix']), data['num_vehicles'], data['depot']
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def emission_callback(from_index, to_index):
        """Returns the emission between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(data['emission_matrix'][from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(emission_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # no slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity'
    )

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
        print("No solution found.")

if __name__ == "__main__":
    main()
