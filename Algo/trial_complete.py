import geopandas as gpd
import numpy as np
import requests
import pandas as pd
import folium
from pyvrp import Model
from pyvrp.plotting import plot_solution
from pyvrp.stop import MaxRuntime
from math import radians, sin, cos, sqrt, atan2

# Compactor location
compactor_location = (23.799829187950845, 86.44036347325256)

# Load the shapefiles
shapefiles = ['clean_26.shp', 'final ward 27 intersection point.shp', 'Ward30_Finalpoints.shp']
coordinates = []

for shapefile in shapefiles:
    wards_shp = gpd.read_file(shapefile)
    wards_shp = wards_shp[wards_shp['geometry'].notnull()]
    wards_shp['coords'] = wards_shp['geometry'].apply(
        lambda geom: (geom.y, geom.x) if geom is not None else None
    )
    coordinates.extend(wards_shp['coords'].tolist())

# Add the compactor location to the coordinates list
coordinates.insert(0, compactor_location)

# Initialize demands (1 for each bin location)
DEMANDS = [1] * len(coordinates)

# Function to calculate Haversine distance between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c * 1000  # Return distance in meters (positive value)
    return distance

# Step 1: Scale the latitude and longitude to integers (multiply by a scale factor)
SCALE_FACTOR = 10000
scaled_coords = [(int(lat * SCALE_FACTOR), int(lon * SCALE_FACTOR)) for lat, lon in coordinates]

# Model setup
m = Model()
m.add_vehicle_type(13, capacity=10)  # 13 trucks, capacity of 10

# Add depot
depot = m.add_depot(x=scaled_coords[0][0], y=scaled_coords[0][1])

# Add clients
clients = [m.add_client(x=scaled_coords[idx][0], y=scaled_coords[idx][1], delivery=DEMANDS[idx]) for idx in range(1, len(coordinates))]
locations = [depot] + clients

# Add edges using Haversine distance
for frm in locations:
    for to in locations:
        # Use the original lat/lon values for Haversine distance calculation
        lat1, lon1 = coordinates[locations.index(frm)]
        lat2, lon2 = coordinates[locations.index(to)]
        
        # Calculate Haversine distance, convert to integer and ensure it's positive
        distance = abs(int(haversine(lat1, lon1, lat2, lon2)))
        m.add_edge(frm, to, distance=distance)

# Solve the problem with a 1-second runtime limit
res = m.solve(stop=MaxRuntime(1), display=True)

# Check if a valid solution was found and visualize using Folium
if res.best:
    # Create a Folium map centered at the compactor location
    route_map = folium.Map(location=compactor_location, zoom_start=14)

    # Add marker for compactor location (depot)
    folium.Marker(compactor_location, popup="Compactor Location", icon=folium.Icon(color="red")).add_to(route_map)

    # Add markers for client locations (bins)
    for idx, coord in enumerate(coordinates[1:], start=1):
        folium.Marker(coord, popup=f"Bin {idx}", icon=folium.Icon(color="green")).add_to(route_map)

    # Plot the solution routes on the map
    for route in res.best.routes():
        latlons = [coordinates[idx] for idx in route]
        folium.PolyLine(latlons, color="blue", weight=2.5, opacity=1).add_to(route_map)

    # Save the map to an HTML file and display it
    route_map.save('optimized_routes_map.html')
    print("Optimal routes saved to 'optimized_routes_map.html'")
else:
    print("No feasible solution was found.")
