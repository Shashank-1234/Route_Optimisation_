import matplotlib.pyplot as plt
from pyvrp import Model
from pyvrp.plotting import plot_solution
from pyvrp.stop import MaxRuntime
from math import radians, sin, cos, sqrt, atan2
import nextplot as nxp  # Import Nextplot for real-time map visualization
import folium
import requests


# Function to calculate Haversine distance between two lat/lon points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c * 1000  # Return distance in meters (positive value)
    return distance


# Example Latitude/Longitude Coordinates for London area (depot and clients)
COORDS = [
    (51.5074, -0.1278),  # Depot: London (latitude, longitude)
    (51.5155, -0.1426),  # Client 1
    (51.5033, -0.1195),  # Client 2
    (51.5145, -0.1045),  # Client 3
    (51.5126, -0.1389),  # Client 4
    (51.5185, -0.0813),  # Client 5
    (51.5201, -0.0965),  # Client 6
    (51.5134, -0.0876),  # Client 7
    (51.5271, -0.1040),  # Client 8
    (51.5080, -0.1282),  # Client 9
    (51.5200, -0.1449),  # Client 10
    (51.5225, -0.1341),  # Client 11
    (51.5098, -0.1180),  # Client 12
    (51.5060, -0.1290),  # Client 13
    (51.5039, -0.1248),  # Client 14
    (51.5116, -0.0880),  # Client 15
    (51.5188, -0.1121),  # Client 16
]

DEMANDS = [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]  # Example demands

# Step 1: Scale the latitude and longitude to integers (multiply by a scale factor)
SCALE_FACTOR = 10000
scaled_coords = [(int(lat * SCALE_FACTOR), int(lon * SCALE_FACTOR)) for lat, lon in COORDS]

# Model setup
m = Model()
m.add_vehicle_type(4, capacity=15)

# Add depot
depot = m.add_depot(x=scaled_coords[0][0], y=scaled_coords[0][1])

# Add clients
clients = [m.add_client(x=scaled_coords[idx][0], y=scaled_coords[idx][1], delivery=DEMANDS[idx]) for idx in range(1, len(COORDS))]
locations = [depot] + clients

# Add edges using Haversine distance
for frm in locations:
    for to in locations:
        # Use the original lat/lon values for Haversine distance calculation
        lat1, lon1 = COORDS[locations.index(frm)]
        lat2, lon2 = COORDS[locations.index(to)]
        
        # Calculate Haversine distance, convert to integer and ensure it's positive
        distance = abs(int(haversine(lat1, lon1, lat2, lon2)))
        m.add_edge(frm, to, distance=distance)

# Solve the problem with a 1-second runtime limit
res = m.solve(stop=MaxRuntime(1), display=True)

# Function to plot routes with Nextplot
def plot_routes_with_nextplot(solution, coords):
    """
    Plots the optimized routes using Nextplot.
    :param solution: Solution object from PyVRP
    :param coords: List of (latitude, longitude) coordinates
    """
    
    plot = nxp.Plot(title="Optimized VRP Routes", size=(800, 800))

    for route in solution.routes():
        latlons = [(coords[idx][0], coords[idx][1]) for idx in route]
        plot.route(latlons, color="blue")  # Plot the route
    
    # Add depot and clients as markers
    plot.marker(coords[0], color="red", label="Depot")  # Depot in red
    for idx, coord in enumerate(coords[1:], start=1):
        plot.marker(coord, color="green", label=f"Client {idx}")  # Clients in green

    plot.show()  # Display the plot

# Function to plot routes on a Folium map
def plot_routes_on_folium(solution, coords):
    """
    Plots the optimized routes on a Folium map.
    :param solution: Solution object from PyVRP
    :param coords: List of (latitude, longitude) coordinates
    """
    # Initialize the map centered at the depot
    folium_map = folium.Map(location=coords[0], zoom_start=13)

    # Add markers for depot and clients
    folium.Marker(coords[0], popup="Depot", icon=folium.Icon(color='red')).add_to(folium_map)
    for idx, coord in enumerate(coords[1:], start=1):
        folium.Marker(coord, popup=f"Client {idx}", icon=folium.Icon(color='green')).add_to(folium_map)

    # Plot each route as a polyline
    for route in solution.routes():
        route_coords = [coords[idx] for idx in route]
        folium.PolyLine(route_coords, color="blue", weight=2.5, opacity=1).add_to(folium_map)

    # Display the map
    return folium_map

# Check if a valid solution was found
if res.best:
    # Visualize with Nextplot
    plot_routes_with_nextplot(res.best, COORDS)
    
    # Visualize on Folium map
    folium_map = plot_routes_on_folium(res.best, COORDS)
    folium_map.save("vrp_solution_map.html")  # Save the map to an HTML file
else:
    print("No feasible solution was found.")
