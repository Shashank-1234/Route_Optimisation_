
import geopandas as gpd
import numpy as np
import pandas as pd
from pyvrp import Model
from pyvrp.stop import MaxIterations
import folium
import requests
import matplotlib.pyplot as plt
from pyvrp.plotting import plot_result


# Load the distance and time matrices from the Excel file
excel_file = 'Algo/gmaps_distance_time_matrix_pooled.xlsx'

# Read distance and time matrices from respective sheets
distance_matrix = pd.read_excel(excel_file, sheet_name='Distance_Matrix', index_col=0)
time_matrix = pd.read_excel(excel_file, sheet_name='Time_Matrix', index_col=0)

# Convert distance from meters to miles
distance_matrix_miles = distance_matrix * 0.000621371

# Convert time from seconds to hours
time_matrix_hours = time_matrix / 3600

# Calculate speed in miles per hour (mph)
speed_matrix = distance_matrix_miles / time_matrix_hours

# Parameters for CO2 emissions
A, B = -0.025952, 0.000309
baseline_emission_factor = 0.76239  # g/m for BS-II trucks from the CPCB report

# Function to calculate emissions
def calculate_emissions(distance, speed):
    CCF = np.exp(A * (speed - 27.4) + B * (speed - 27.4)**2)
    return baseline_emission_factor * CCF * distance

# Create an empty emission matrix
emission_matrix = pd.DataFrame(index=distance_matrix.index, columns=distance_matrix.columns)

# Calculate the emissions for each route segment
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

# Load shapefiles and extract coordinates for Ward 26 and Ward 27 (with demand = 1)
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
    # Add demand of 1 for these coordinates
    DEMANDS.extend([1] * len(wards_shp))

# Load Ward 30 shapefile with demand = 2
ward_30_shp = gpd.read_file('Algo/Ward30_Finalpoints.shp')
ward_30_shp = ward_30_shp[ward_30_shp['geometry'].notnull()]
ward_30_shp['coords'] = ward_30_shp['geometry'].apply(
    lambda geom: (geom.y, geom.x) if geom is not None else None
)
coordinates.extend(ward_30_shp['coords'].tolist())
# Add demand of 2 for these coordinates
DEMANDS.extend([2] * len(ward_30_shp))

# Add the compactor location (depot) to the coordinates list
coordinates.insert(0, compactor_location)
DEMANDS.insert(0, 0)  # The depot should have a demand of 0

# Initialize the model
m = Model()
m.add_vehicle_type(
    num_available=18,
    capacity=10,
)

# Add depot
depot = m.add_depot(x=coordinates[0][1] * 10000, y=coordinates[0][0] * 10000)

# Add clients
clients = [m.add_client(x=coordinates[idx][1] * 10000, y=coordinates[idx][0] * 10000, delivery=DEMANDS[idx]) for idx in range(1, len(coordinates))]
locations = [depot] + clients

# Add a profile to handle both distance and duration
profile = m.add_profile()

# Use the emission matrix as the cost matrix for pyvrp (instead of distance)
for frm_idx, frm in enumerate(locations):
    for to_idx, to in enumerate(locations):
        if frm_idx == to_idx:
            emission_cost = 0  # Avoid self-loops
        else:
            emission_cost = emission_matrix.iloc[frm_idx, to_idx]
        
        # Add edge with emission as the cost, keep float for accuracy
        m.add_edge(frm, to, distance= float(emission_cost), profile=profile)

# Solve the problem
res = m.solve(stop=MaxIterations(1000), seed=15)

# Initialize total values for emissions, distance, and duration
total_emissions_all_routes = 0
total_distance_all_routes = 0
total_duration_all_routes = 0

for vehicle_idx, route in enumerate(res.best.routes()):
    total_emissions = 0
    total_distance = 0
    total_duration = 0
    route_sequence = []  # Track sequence of stops for each vehicle

    for i in range(len(route)):
        frm_idx = 0 if i == 0 else route[i - 1]
        to_idx = route[i]

        # Fetch distance and duration from matrices
        distance = distance_matrix.iloc[frm_idx, to_idx]
        duration = time_matrix.iloc[frm_idx, to_idx]  # Duration (in seconds)

        # Fetch speed from the speed matrix
        speed = speed_matrix.iloc[frm_idx, to_idx]

        # Calculate emissions
        emissions = calculate_emissions(distance, speed)
        total_emissions += emissions

        # Add distance and duration
        total_distance += distance
        total_duration += duration

        # Append to route sequence
        route_sequence.append(to_idx)

    # Print for the current vehicle
    print(f"Vehicle {vehicle_idx + 1}:")
    print(f"  Route sequence: {route_sequence}")
    print(f"  Total emissions: {total_emissions:.2f} g")
    print(f"  Total distance: {total_distance:.2f} meters")
    print(f"  Total duration: {total_duration:.2f} seconds")

    # Add to the overall totals
    total_emissions_all_routes += total_emissions
    total_distance_all_routes += total_distance
    total_duration_all_routes += total_duration

# Print totals for all routes
print(f"Total emissions for all routes: {total_emissions_all_routes:.2f} g")
print(f"Total distance for all routes: {total_distance_all_routes:.2f} meters")
print(f"Total duration for all routes: {total_duration_all_routes:.2f} seconds")


"""
import geopandas as gpd
import numpy as np
import pandas as pd
from pyvrp import Model
from pyvrp.stop import MaxIterations
import folium

# Load the distance and time matrices from the Excel file
excel_file = 'Algo/gmaps_distance_time_matrix_pooled.xlsx'
distance_matrix = pd.read_excel(excel_file, sheet_name='Distance_Matrix', index_col=0)
time_matrix = pd.read_excel(excel_file, sheet_name='Time_Matrix', index_col=0)

# Convert distance from meters to miles
distance_matrix_miles = distance_matrix * 0.000621371
time_matrix_hours = time_matrix / 3600

# Calculate speed in miles per hour (mph)
speed_matrix = distance_matrix_miles / time_matrix_hours

# Emission coefficients for different pollutants
coefficients = {
    "CO2": {"A": -0.025952, "B": 0.000309},
    "CO": {"A": -0.028971, "B": 0.001922},
    "NOx": {"A": 0.008967, "B": -0.000027},
    "HC": {"A": -0.031762, "B": 0.000908},
}

# Baseline emission factors for different pollutants (g/m)
baseline_factors = {
    "CO2": 0.76239,
    "CO": 0.0100,  # Example value; replace with actual value
    "NOx": 0.0025,  # Example value; replace with actual value
    "HC": 0.0010,  # Example value; replace with actual value
}

# Define weights for each pollutant in the optimization
weights = {
    "CO2": 1.0,   # Adjust these weights based on importance
    "CO": 0.5,    # For example, CO is half as important as CO2
    "NOx": 2.0,   # NOx is twice as important as CO2
    "HC": 0.7     # HC is 0.7 times the importance of CO2
}

# Function to calculate emissions for each pollutant
def calculate_emissions(distance, speed, A, B, baseline_factor):
    CCF = np.exp(A * (speed - 27.4) + B * (speed - 27.4)**2)
    return baseline_factor * CCF * distance

# Create emission matrices for each pollutant
emission_matrices = {pollutant: pd.DataFrame(index=distance_matrix.index, columns=distance_matrix.columns) for pollutant in coefficients}

# Calculate the emissions for each pollutant and route segment
for pollutant, params in coefficients.items():
    A = params["A"]
    B = params["B"]
    baseline_factor = baseline_factors[pollutant]

    for i in distance_matrix.index:
        for j in distance_matrix.columns:
            distance = distance_matrix.loc[i, j]
            speed = speed_matrix.loc[i, j]
            if distance > 0 and speed > 0:  # Avoid self-loops
                emission_matrices[pollutant].loc[i, j] = calculate_emissions(distance, speed, A, B, baseline_factor)
            else:
                emission_matrices[pollutant].loc[i, j] = 0

# Create a combined cost matrix using a weighted sum of all pollutant emissions
combined_cost_matrix = pd.DataFrame(index=distance_matrix.index, columns=distance_matrix.columns)

for i in distance_matrix.index:
    for j in distance_matrix.columns:
        combined_cost = 0
        for pollutant in coefficients:
            combined_cost += weights[pollutant] * emission_matrices[pollutant].loc[i, j]
        combined_cost_matrix.loc[i, j] = combined_cost

# Compactor (depot) location and shapefiles loading as before
compactor_location = (23.799829187950845, 86.44036347325256)
shapefiles = ['Algo/clean_26.shp', 'Algo/final ward 27 intersection point.shp']
coordinates = []
DEMANDS = []

for shapefile in shapefiles:
    wards_shp = gpd.read_file(shapefile)
    wards_shp = wards_shp[wards_shp['geometry'].notnull()]
    wards_shp['coords'] = wards_shp['geometry'].apply(lambda geom: (geom.y, geom.x) if geom is not None else None)
    coordinates.extend(wards_shp['coords'].tolist())
    DEMANDS.extend([1] * len(wards_shp))

ward_30_shp = gpd.read_file('Algo/Ward30_Finalpoints.shp')
ward_30_shp = ward_30_shp[ward_30_shp['geometry'].notnull()]
ward_30_shp['coords'] = ward_30_shp['geometry'].apply(lambda geom: (geom.y, geom.x) if geom is not None else None)
coordinates.extend(ward_30_shp['coords'].tolist())
DEMANDS.extend([2] * len(ward_30_shp))

# Add depot location
coordinates.insert(0, compactor_location)
DEMANDS.insert(0, 0)

# Initialize the model
m = Model()
m.add_vehicle_type(num_available=18, capacity=10)
depot = m.add_depot(x=coordinates[0][1] * 10000, y=coordinates[0][0] * 10000)
clients = [m.add_client(x=coordinates[idx][1] * 10000, y=coordinates[idx][0] * 10000, delivery=DEMANDS[idx]) for idx in range(1, len(coordinates))]
locations = [depot] + clients
profile = m.add_profile()

# Use the combined cost matrix as the cost function for pyvrp
for frm_idx, frm in enumerate(locations):
    for to_idx, to in enumerate(locations):
        if frm_idx == to_idx:
            combined_cost = 0
        else:
            combined_cost = combined_cost_matrix.iloc[frm_idx, to_idx]
        
        m.add_edge(frm, to, distance=float(combined_cost), profile=profile)

# Solve the problem
res = m.solve(stop=MaxIterations(2000), seed=42)
"""

# OSRM server URL (make sure your server is running and available at this URL)
OSRM_URL = "http://localhost:5001/route/v1/driving"

# Function to get the OSRM route between two locations
def get_osrm_route(origin, destination):
    query = f"{OSRM_URL}/{origin[1]},{origin[0]};{destination[1]},{destination[0]}?geometries=geojson&overview=full"
    response = requests.get(query)
    data = response.json()
    
    if 'routes' in data and data['routes']:
        geometry = data['routes'][0]['geometry']['coordinates']
        distance = data['routes'][0]['distance']
        return geometry, distance
    else:
        print(f"Error: No route found between {origin} and {destination}")
        return None, None

# Predefined list of colors for each vehicle route
colors = [
    "blue", "red", "green", "purple", "orange", "darkred", "lightred", "black", "darkblue", 
    "darkgreen", "cadetblue", "pink", "lightblue"
]

# Check if a valid solution was found and visualize using Folium
if res.best:
    # Create a Folium map centered at the compactor location
    route_map = folium.Map(location=compactor_location, zoom_start=14)

    # Add marker for compactor location (depot)
    folium.Marker(compactor_location, popup="Compactor Location", icon=folium.Icon(color="red")).add_to(route_map)

    # Add markers for client locations (bins)
    for idx, coord in enumerate(coordinates[1:], start=1):
        folium.Marker(coord, popup=f"Bin {idx}", icon=folium.Icon(color="green")).add_to(route_map)

    # Plot the solution routes using real OSRM routes
    for vehicle_idx, route in enumerate(res.best.routes()):
        # Start the route at the depot
        latlons = [coordinates[0]]  # Starting from the depot
        # Append all bins the vehicle visits
        for i in route:
            latlons.append(coordinates[i])
        # Return to the depot
        latlons.append(coordinates[0])

        # Get the color for this vehicle route
        color = colors[vehicle_idx % len(colors)]  # Ensure we cycle through the color list

        # Ensure that we request the full route at once (from depot -> bins -> depot)
        for i in range(len(latlons) - 1):
            latlon1 = latlons[i]
            latlon2 = latlons[i + 1]
            route_geometry, _ = get_osrm_route(latlon1, latlon2)
            if route_geometry:
                # Ensure latitude comes first, then longitude
                route_geometry_latlon = [(coord[1], coord[0]) for coord in route_geometry]
                
                # Adding the route line with tooltips and highlight function
                folium.PolyLine(
                    route_geometry_latlon,
                    color=color,
                    weight=2.5,
                    opacity=1,
                    tooltip=f"Vehicle {vehicle_idx + 1}",  # Tooltip showing the vehicle index
                    highlight_function=lambda x: {'color': 'black', 'weight': 5},  # Change color to black and weight to 5 on hover
                    line_cap='round'  # Ensures smooth line caps when enlarged
                ).add_to(route_map)


    # Save the map to an HTML file after processing all routes
    route_map.save('optimized_routes_osrm_map.html')
    print("Optimal routes saved to 'optimized_routes_osrm_map.html'")


data = m.data  # Check if `m.data` is correct or directly use your ProblemData instance here

# Plot the result
fig, ax = plt.subplots(figsize=(10, 8))
plot_result(res, data, fig=fig)
plt.show()