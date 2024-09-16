import pandas as pd
import numpy as np
import random
from deap import base, creator, tools
import matplotlib.pyplot as plt

# Load the distance and time matrices from Excel
file_path = "Algo/gmaps_distance_time_matrix_pooled.xlsx"
distance_matrix_df = pd.read_excel(file_path, sheet_name="Distance_Matrix")
time_matrix_df = pd.read_excel(file_path, sheet_name="Time_Matrix")

# Convert to numpy arrays (without index column)
distance_matrix = distance_matrix_df.drop(columns=['Unnamed: 0']).values
time_matrix = time_matrix_df.drop(columns=['Unnamed: 0']).values

# Ensure the time matrix is in minutes (assuming it's in seconds)
time_matrix = time_matrix / 60  # Convert seconds to minutes if needed

# Set Parameters and Constraints
n = distance_matrix.shape[0]  # Number of cities (including depot)
working_hours = 8 * 60  # Max working minutes per day (8 hours)
service_time_per_bin = 5  # Service time per bin (minutes)
unloading_time_per_round = 30  # Unloading time per round (minutes)
max_bins_per_round = 11  # Max bins per round
total_break_time = 30  # Total break time for the day

# Genetic Algorithm parameters
population_size = 50
generations = 100
crossover_probability = 0.8
mutation_probability = 0.2
tournament_size = 3

# DEAP setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))  # Minimize distance and time
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Individual creation (random route visiting all bins)
def create_individual():
    return [0] + random.sample(range(1, n), n - 1) + [0]  # Start and end at depot

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register selection, crossover, and mutation functions
toolbox.register("select", tools.selTournament, tournsize=tournament_size)
toolbox.register("mate", tools.cxOrdered)  # Sequential Constructive Crossover (SCX)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)  # Swap mutation

# Step 3: Calculate travel time in minutes based on time matrix
def calculate_travel_time(route):
    return sum(time_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

# Split route for multiple rounds and two trucks
def split_route(individual):
    sub_routes_truck1 = []
    sub_routes_truck2 = []

    visited_bins = set()  # Track visited bins
    current_truck1_route = [0]  # Start at depot for truck 1
    current_truck2_route = [0]  # Start at depot for truck 2

    current_truck1_time = 0
    current_truck2_time = 0
    current_truck1_bins = 0
    current_truck2_bins = 0

    for next_city in individual[1:-1]:
        if next_city in visited_bins:
            continue

        # Try to assign bins to Truck 1
        travel_time_to_next = time_matrix[current_truck1_route[-1]][next_city]
        if current_truck1_bins < max_bins_per_round and current_truck1_time + travel_time_to_next + service_time_per_bin <= working_hours:
            current_truck1_route.append(next_city)
            current_truck1_time += travel_time_to_next + service_time_per_bin
            current_truck1_bins += 1
            visited_bins.add(next_city)
        # If Truck 1 is full or exceeds time, assign to Truck 2
        else:
            travel_time_to_next = time_matrix[current_truck2_route[-1]][next_city]
            if current_truck2_bins < max_bins_per_round and current_truck2_time + travel_time_to_next + service_time_per_bin <= working_hours:
                current_truck2_route.append(next_city)
                current_truck2_time += travel_time_to_next + service_time_per_bin
                current_truck2_bins += 1
                visited_bins.add(next_city)
    
    # If there are still bins unvisited and both trucks are full, return to the depot
    current_truck1_route.append(0)
    current_truck2_route.append(0)

    sub_routes_truck1.append(current_truck1_route)
    sub_routes_truck2.append(current_truck2_route)

    return sub_routes_truck1, sub_routes_truck2

# Fitness evaluation
def evaluate_individual(individual):
    sub_routes_truck1, sub_routes_truck2 = split_route(individual)

    total_distance_truck1 = sum(distance_matrix[route[i]][route[i + 1]] for route in sub_routes_truck1 for i in range(len(route) - 1))
    total_time_truck1 = sum(calculate_travel_time(route) for route in sub_routes_truck1)

    total_distance_truck2 = sum(distance_matrix[route[i]][route[i + 1]] for route in sub_routes_truck2 for i in range(len(route) - 1))
    total_time_truck2 = sum(calculate_travel_time(route) for route in sub_routes_truck2)

    combined_fitness = (total_distance_truck1 + total_distance_truck2), (total_time_truck1 + total_time_truck2)
    return combined_fitness

toolbox.register("evaluate", evaluate_individual)

# Genetic Algorithm Execution
def run_ga_with_immigration(runs=10, generations=100, pop_size=50, immigration_rate=0.05):
    best_solutions = []
    for run in range(runs):
        population = toolbox.population(n=pop_size)

        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        for gen in range(generations):
            selected = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, selected))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_probability:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

                if random.random() < mutation_probability:
                    toolbox.mutate(child1)
                    toolbox.mutate(child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)

            population[:] = offspring

        best_individual = tools.selBest(population, 1)[0]
        best_solutions.append(best_individual)

    return best_solutions

# Run the genetic algorithm
best_solutions = run_ga_with_immigration()

# Function to print the best solution in the desired format
def print_best_solution(best_solution):
    sub_routes_truck1, sub_routes_truck2 = split_route(best_solution)

    total_time_truck1 = sum(calculate_travel_time(route) + len(route) * service_time_per_bin + unloading_time_per_round for route in sub_routes_truck1)
    total_time_truck2 = sum(calculate_travel_time(route) + len(route) * service_time_per_bin + unloading_time_per_round for route in sub_routes_truck2)

    total_time = total_time_truck1 + total_time_truck2

    print("\n--- Best Solution ---\n")
    print("Truck 1 Routes:")
    for i, route in enumerate(sub_routes_truck1):
        route_str = ' -> '.join(map(str, route))
        route_time = calculate_travel_time(route) + len(route) * service_time_per_bin + unloading_time_per_round
        print(f"Round {i + 1}: {route_str} | Time: {route_time} minutes")

    print("\nTruck 2 Routes:")
    for i, route in enumerate(sub_routes_truck2):
        route_str = ' -> '.join(map(str, route))
        route_time = calculate_travel_time(route) + len(route) * service_time_per_bin + unloading_time_per_round
        print(f"Round {i + 1}: {route_str} | Time: {route_time} minutes")

    print(f"\nTotal Time for Truck 1: {total_time_truck1} minutes")
    print(f"Total Time for Truck 2: {total_time_truck2} minutes")
    print(f"Overall Total Time: {total_time} minutes")

# Get the best solution after running the genetic algorithm
best_solution = tools.selBest(best_solutions, 1)[0]


# Print the best solution in the requested format
print_best_solution(best_solution)
