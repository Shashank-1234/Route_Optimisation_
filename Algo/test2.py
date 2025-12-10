import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

# Load the distance and time matrices from Excel
file_path = "Algo/gmaps_distance_time_matrix_pooled.xlsx"

# Load the matrices, skipping the first row and first column to get rid of the labels
distance_matrix_df = pd.read_excel(file_path, sheet_name="Distance_Matrix", index_col=0)
time_matrix_df = pd.read_excel(file_path, sheet_name="Time_Matrix", index_col=0)

# Ensure that the matrices are numeric by converting the data to numeric values
distance_matrix_df = distance_matrix_df.apply(pd.to_numeric, errors='coerce')
time_matrix_df = time_matrix_df.apply(pd.to_numeric, errors='coerce')

# Convert DataFrame to NumPy arrays
distance_matrix = distance_matrix_df.values
time_matrix = time_matrix_df.values

# Parameters
num_points = distance_matrix.shape[0]  # Number of points (including depot)
num_trucks = 2  # Number of trucks
max_bins_per_route = 10  # Maximum number of bins per route
max_time_per_route = 8 * 3600  # 8 hours in seconds
service_time_per_bin = 5 * 60  # 5 minutes in seconds
break_time_per_shift = 30 * 60  # 30 minutes in seconds

# DEAP setup
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # Minimize both objectives
creator.create("Individual", list, fitness=creator.FitnessMulti)

def create_individual():
    """Create an individual with random routes for each truck."""
    individual = []
    for _ in range(num_trucks):
        route = random.sample(range(1, num_points), max_bins_per_route)  # Randomly assign bins
        individual.append(route)
    return creator.Individual(individual)

def evaluate(individual):
    """Evaluate the individual based on total distance and time constraints."""
    total_distance = 0
    total_time = 0

    for truck_route in individual:
        if not truck_route:  # If no route, skip
            continue
        # Add depot as start and end point (0)
        truck_route = [0] + truck_route + [0]
        
        route_distance = 0
        route_time = 0
        for i in range(len(truck_route) - 1):
            route_distance += distance_matrix[truck_route[i], truck_route[i + 1]]
            route_time += time_matrix[truck_route[i], truck_route[i + 1]]

        # Add service time per bin and break time if applicable
        route_time += len(truck_route[1:-1]) * service_time_per_bin  # Ignore depot in service time
        route_time += break_time_per_shift  # Break per shift
        
        # Update total distance and time
        total_distance += route_distance
        total_time += route_time

    # Objective 1: Minimize total distance
    # Objective 2: Minimize time violation (penalty if exceeding 8 hours)
    time_violation = max(0, total_time - max_time_per_route)
    return total_distance, time_violation

def crossover(ind1, ind2):
    """Crossover between two individuals (exchange routes between trucks)."""
    truck_idx = random.randint(0, num_trucks - 1)  # Choose a random truck
    ind1[truck_idx], ind2[truck_idx] = ind2[truck_idx], ind1[truck_idx]  # Swap routes
    return ind1, ind2

def mutate(individual):
    """Mutate an individual by swapping two random bins between routes."""
    truck_idx = random.randint(0, num_trucks - 1)
    route = individual[truck_idx]
    if len(route) >= 2:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return individual,

# Genetic Algorithm setup
toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selNSGA2)

# Parameters
population_size = 100
num_generations = 100
crossover_probability = 0.7
mutation_probability = 0.2

def main():
    population = toolbox.population(n=population_size)
    algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size,
                              cxpb=crossover_probability, mutpb=mutation_probability, ngen=num_generations,
                              stats=None, halloffame=None, verbose=True)
    
    # Get the Pareto front
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    return pareto_front

if __name__ == "__main__":
    pareto_front = main()
    for ind in pareto_front:
        print(f"Route: {ind}, Fitness: {ind.fitness.values}")
