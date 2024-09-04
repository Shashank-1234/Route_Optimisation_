import geopandas as gpd
import numpy as np
import requests
import pandas as pd

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

# Initialize distance matrix
n = len(coordinates)
distance_matrix = np.zeros((n, n))

# Function to get route distance using OSRM
def get_route_distance(origin, destination):
    osrm_url = f"http://127.0.0.1:5001/route/v1/driving/{origin[1]},{origin[0]};{destination[1]},{destination[0]}?overview=false"
    response = requests.get(osrm_url)
    data = response.json()
    if data['routes']:
        return data['routes'][0]['distance']  # Distance in meters
    else:
        print(f"No valid route found between {origin} and {destination}.")
        return None

# Populate the distance matrix
for i, origin in enumerate(coordinates):
    for j, destination in enumerate(coordinates):
        if i != j:
            distance = get_route_distance(origin, destination)
            if distance is not None:
                distance_matrix[i][j] = distance

# Convert the distance matrix to a DataFrame
distance_df = pd.DataFrame(distance_matrix, columns=[f'Point_{i}' for i in range(n)], index=[f'Point_{i}' for i in range(n)])

# Export the distance matrix to an Excel file
distance_df.to_excel('distance_matrix.xlsx', index=True)

# Print confirmation
print("Distance matrix exported to 'distance_matrix.xlsx'")
