import pandas as pd
import numpy as np
import random
from deap import base, creator, tools
import matplotlib.pyplot as plt
from statistics import mean, stdev

# Step 1: Load the distance matrix from Excel
file_path = "Algo/distance_matrix.xlsx"  # Update the path if necessary
distance_matrix_df = pd.read_excel(file_path)

# Convert to numpy array (without index column)
distance_matrix = distance_matrix_df.drop(columns=['Unnamed: 0']).values

# Parameters and Constraints
n = distance_matrix.shape[0]  # Number of cities/bins (including depot)
working_hours = 4 * 60  # Max working minutes per trip (4 hours)
speed_kmh = 25  # Speed in km/h
service_time_per_bin = 10  # Service time per bin (minutes)
max_bins_per_route = 10  # Bin limit per truck route

# Convert distance matrix from meters to kilometers
distance_matrix_km = distance_matrix / 1000

# DEAP setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Individual creation (random route visiting all bins)
def create_individual():
    return [0] + random.sample(range(1, n), n - 1) + [0]  # Start and end at depot

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Step 3: Calculate travel time in minutes
def calculate_travel_time_km(distance_km):
    return (distance_km / speed_kmh) * 60  # Convert hours to minutes

# Split route with time and bin constraints
def split_route(individual):
    sub_routes = []
    current_route = [0]  # Start at depot
    current_time = 0
    current_bin_count = 0

    for i in range(1, len(individual) - 1):
        next_city = individual[i]
        travel_time_to_next = calculate_travel_time_km(distance_matrix_km[current_route[-1]][next_city])

        # Check total route time within 8 hours and bin limit
        if (current_time + travel_time_to_next + service_time_per_bin <= working_hours) and (current_bin_count < max_bins_per_route):
            current_route.append(next_city)
            current_time += travel_time_to_next + service_time_per_bin
            current_bin_count += 1
        else:
            current_route.append(0)  # End at depot
            sub_routes.append(current_route)
            current_route = [0, next_city]  # Start new route
            current_time = calculate_travel_time_km(distance_matrix_km[0][next_city]) + service_time_per_bin
            current_bin_count = 1

    current_route.append(0)  # End the last route at the depot
    sub_routes.append(current_route)
    return sub_routes

# Fitness evaluation
def route_distance(route):
    return sum(distance_matrix_km[route[i]][route[i+1]] for i in range(len(route) - 1))

def total_distance_of_sub_routes(sub_routes):
    return sum(route_distance(route) for route in sub_routes)

def evaluate_individual(individual):
    sub_routes = split_route(individual)
    return total_distance_of_sub_routes(sub_routes),

toolbox.register("evaluate", evaluate_individual)

# Step 6: Local search (2-opt), crossover, and mutation
def two_opt(route):
    best_route = route[:]
    best_distance = route_distance(best_route)
    
    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):
            new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
            new_distance = route_distance(new_route)
            if new_distance < best_distance:
                best_route = new_route
                best_distance = new_distance
    return best_route

def apply_two_opt_to_sub_routes(sub_routes):
    return [two_opt(route) for route in sub_routes]

# Crossover and mutation functions
def scx_crossover(parent1, parent2):
    offspring = [0]
    current_city = 0

    for i in range(1, n):
        parent1_next_city = next_city_in_parent(parent1, current_city)
        parent2_next_city = next_city_in_parent(parent2, current_city)

        if distance_matrix_km[current_city][parent1_next_city] <= distance_matrix_km[current_city][parent2_next_city]:
            offspring.append(parent1_next_city)
            current_city = parent1_next_city
        else:
            offspring.append(parent2_next_city)
            current_city = parent2_next_city

    return offspring

def next_city_in_parent(parent, current_city):
    for city in parent:
        if city != current_city and city != -1:
            return city
    return -1

def exchange_mutation(individual):
    idx1, idx2 = random.sample(range(1, len(individual) - 1), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]  
    return individual


# Register crossover, mutation, and selection
toolbox.register("select", tools.selTournament, tournsize=4)  # Increased selection pressure
toolbox.register("mate", scx_crossover)
toolbox.register("mutate", exchange_mutation)

# Step 7: Genetic algorithm with elitism, immigration, and adaptive mutation
def run_ga(generations=300, pop_size=200, immigration_rate=0.05, stagnation_limit=50):
    population = toolbox.population(n=pop_size)
    distance_history = []

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    best_distance_overall = float('inf')
    stagnation_counter = 0  # Counter to track stagnation
    mutation_probability = 0.05  # Start with a low mutation rate

    for gen in range(generations):
        # Selection and cloning
        selected = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, selected))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.8:  # Crossover probability
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

            # Dynamic mutation rate adjustment
            if random.random() < mutation_probability:  
                toolbox.mutate(child1)
                toolbox.mutate(child2)
                del child1.fitness.values, child2.fitness.values

        # Evaluate the offspring
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # Apply elitism
        best_individuals = tools.selBest(population, 5)  # Keep the best 5 individuals
        population[:5] = best_individuals
        population[:] = offspring

        # Record best distance for this generation
        best_individual = tools.selBest(population, 1)[0]
        best_distance = evaluate_individual(best_individual)[0]
        distance_history.append(best_distance)

        # Check for stagnation
        if best_distance < best_distance_overall:
            best_distance_overall = best_distance
            stagnation_counter = 0  # Reset stagnation counter if improvement found

            # Reduce mutation rate if improving
            if mutation_probability > 0.01:
                mutation_probability *= 0.9  # Gradually reduce mutation rate during improvement

        else:
            stagnation_counter += 1

            # Increase mutation rate if stagnation occurs
            if stagnation_counter >= stagnation_limit:
                num_immigrants = int(immigration_rate * pop_size)
                
                # Introduce immigrants by crossing them with the best individual
                for _ in range(num_immigrants):
                    immigrant = toolbox.clone(best_individual)
                    toolbox.mutate(immigrant)  # Mutate immigrant
                    population[random.randint(5, len(population) - 1)] = immigrant

                stagnation_counter = 0  # Reset stagnation counter after immigration

    best_individual = tools.selBest(population, 1)[0]
    best_sub_route = split_route(best_individual)

    return best_sub_route, distance_history

# Run the genetic algorithm and plot results
best_sub_route, distance_history = run_ga(generations=300, pop_size=200)

# Print best routes
for i, route in enumerate(best_sub_route):
    route_str = ' -> '.join(map(str, route))
    print(f"Route for vehicle {i + 1}: {route_str}")

# Plot convergence
plt.plot(distance_history)
plt.title('Convergence of Genetic Algorithm')
plt.xlabel('Generations')
plt.ylabel('Best Distance')
plt.grid(True)
plt.show()
