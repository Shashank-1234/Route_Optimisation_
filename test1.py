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

# Step 2: Set Parameters and Constraints
n = distance_matrix.shape[0]  # Number of cities/bins (including depot)
working_hours = 8 * 60  # Max working minutes per day (8 hours)
speed_kmh = 25  # Speed in km/h
service_time_per_bin = 10  # Service time per bin (minutes)

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

# Step 3: Calculate travel time in minutes based on speed (25 km/h)
def calculate_travel_time_km(distance_km):
    return (distance_km / speed_kmh) * 60  # Convert hours to minutes

# Step 4: Split route with time constraints (working day of 8 hours) and bin limit
def split_route(individual):
    sub_routes = []
    current_route = [0]  # Start at depot
    current_distance = 0
    current_time = 0
    current_bin_count = 0  # Track the number of bins visited in the current route

    # Bin limit per truck route
    max_bins_per_route = 10

    for i in range(1, len(individual) - 1):
        next_city = individual[i]
        distance_to_next = distance_matrix_km[current_route[-1]][next_city]
        travel_time_to_next = calculate_travel_time_km(distance_to_next)

        # Check total route time within 8 working hours (480 minutes)
        # and ensure the bin count doesn't exceed the max limit
        if (current_time + travel_time_to_next + service_time_per_bin <= working_hours) and (current_bin_count < max_bins_per_route):
            current_route.append(next_city)
            current_time += travel_time_to_next + service_time_per_bin
            current_bin_count += 1  # Increment the number of bins visited
        else:
            # End this route and start a new one for the next truck
            current_route.append(0)  # End at depot
            sub_routes.append(current_route)
            
            # Start a new route for the next truck
            current_route = [0, next_city]
            current_time = calculate_travel_time_km(distance_matrix_km[0][next_city]) + service_time_per_bin
            current_bin_count = 1  # Reset the bin count

    current_route.append(0)  # End the last route at the depot
    sub_routes.append(current_route)
    return sub_routes

# Step 5: Fitness evaluation
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

# Register crossover and mutation
toolbox.register("select", tools.selRoulette)
toolbox.register("mate", scx_crossover)
toolbox.register("mutate", exchange_mutation)

# Function to print the routes for each vehicle
def print_routes(sub_routes):
    for i, route in enumerate(sub_routes):
        route_str = ' -> '.join(map(str, route))
        print(f"Route for vehicle {i + 1}: {route_str}")

# Step 7: Run the genetic algorithm with constraints, statistical analysis, and plotting

def run_ga(runs=10, generations=100, pop_size=50):
    best_distances = []
    best_sub_routes = []
    distance_history = []

    for run in range(runs):
        population = toolbox.population(n=pop_size)

        # Evaluate initial population
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        for gen in range(generations):
            selected = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, selected))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values

                toolbox.mutate(child1)
                toolbox.mutate(child2)
                del child1.fitness.values, child2.fitness.values

            # Evaluate the offspring
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)

            # Record best distance for this generation
            best_individual = tools.selBest(population, 1)[0]
            best_distance = evaluate_individual(best_individual)[0]
            distance_history.append(best_distance)

        # Get best individual after all generations
        best_individual = tools.selBest(population, 1)[0]
        best_distance = evaluate_individual(best_individual)[0]
        best_sub_route = split_route(best_individual)
        best_distances.append(best_distance)
        best_sub_routes.append(best_sub_route)

    return best_distances, best_sub_routes, distance_history



def run_ga_with_immigration(runs=10, generations=1000, pop_size=100, immigration_rate=0.05):
    best_distances = []
    best_sub_routes = []
    distance_history = []

    for run in range(runs):
        population = toolbox.population(n=pop_size)

        # Evaluate initial population
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        for gen in range(generations):
            # Selection and cloning
            selected = tools.selTournament(population, len(population), tournsize=3)  # Stronger selection pressure
            offspring = list(map(toolbox.clone, selected))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.8:  # Adjusted crossover probability
                    toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values

                if random.random() < 0.15:  # Adjusted mutation probability
                    toolbox.mutate(child1)
                    toolbox.mutate(child2)
                    del child1.fitness.values, child2.fitness.values

            # Evaluate the offspring
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)

            # Immigration: Introduce new random individuals less frequently
            if gen % 50 == 0:  # Adjusted immigration interval
                num_immigrants = int(immigration_rate * pop_size)
                for _ in range(num_immigrants):
                    immigrant = toolbox.individual()
                    population[random.randint(0, len(population) - 1)] = immigrant

            # Apply elitism: keep the top 5 individuals
            best_individuals = tools.selBest(population, 5)  # Increased elitism
            population[:5] = best_individuals  # Ensure they survive

            # Replace population with the offspring
            population[:] = offspring

            # Record best distance for this generation
            best_individual = tools.selBest(population, 1)[0]
            best_distance = evaluate_individual(best_individual)[0]
            distance_history.append(best_distance)

        # Get best individual after all generations
        best_individual = tools.selBest(population, 1)[0]
        best_distance = evaluate_individual(best_individual)[0]
        best_sub_route = split_route(best_individual)
        best_distances.append(best_distance)
        best_sub_routes.append(best_sub_route)

    return best_distances, best_sub_routes, distance_history

def run_ga_with_adaptive_immigration(runs=10, generations=1000, pop_size=500, immigration_rate=0.02, stagnation_limit=50):
    best_distances = []
    best_sub_routes = []
    distance_history = []

    for run in range(runs):
        population = toolbox.population(n=pop_size)

        # Evaluate initial population
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        best_distance_overall = float('inf')
        stagnation_counter = 0  # Counter to track stagnation

        for gen in range(generations):
            # Selection and cloning
            selected = tools.selTournament(population, len(population), tournsize=3)
            offspring = list(map(toolbox.clone, selected))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.8:  # Crossover probability
                    toolbox.mate(child1, child2)
                    del child1.fitness.values, child2.fitness.values

                if random.random() < 0.05:  # Further reduced mutation probability
                    toolbox.mutate(child1)
                    toolbox.mutate(child2)
                    del child1.fitness.values, child2.fitness.values

            # Evaluate the offspring
            for ind in offspring:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)

            # Apply elitism
            best_individuals = tools.selBest(population, 5)
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
            else:
                stagnation_counter += 1

            # Apply immigration only if stagnation occurs
            if stagnation_counter >= stagnation_limit:
                num_immigrants = int(immigration_rate * pop_size)
                for _ in range(num_immigrants):
                    immigrant = toolbox.individual()
                    population[random.randint(0, len(population) - 1)] = immigrant
                stagnation_counter = 0  # Reset stagnation counter after immigration

        # Get best individual after all generations
        best_individual = tools.selBest(population, 1)[0]
        best_distance = evaluate_individual(best_individual)[0]
        best_sub_route = split_route(best_individual)
        best_distances.append(best_distance)
        best_sub_routes.append(best_sub_route)

    return best_distances, best_sub_routes, distance_history


def run_ga_with_dynamic_mutation_and_refined_immigration(runs=10, generations=1000, pop_size=50, immigration_rate=0.01, stagnation_limit=50, dynamic_mutation=True):
    best_distances = []
    best_sub_routes = []
    distance_history = []

    for run in range(runs):
        population = toolbox.population(n=pop_size)

        # Evaluate initial population
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

        best_distance_overall = float('inf')
        stagnation_counter = 0  # Counter to track stagnation
        mutation_probability = 0.05  # Start with a low mutation rate

        for gen in range(generations):
            # Selection and cloning
            selected = tools.selTournament(population, len(population), tournsize=3)
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
            best_individuals = tools.selBest(population, 10)
            population[:10] = best_individuals
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
                if dynamic_mutation and mutation_probability > 0.01:
                    mutation_probability *= 0.9  # Gradually reduce mutation rate during improvement

            else:
                stagnation_counter += 1

                # Increase mutation rate if stagnation occurs
                if dynamic_mutation and mutation_probability < 0.1:
                    mutation_probability *= 1.1  # Gradually increase mutation rate during stagnation

            # Apply immigration only if stagnation occurs
            if stagnation_counter >= stagnation_limit:
                num_immigrants = int(immigration_rate * pop_size)
                
                # Introduce immigrants by crossing them with the best individual
                for _ in range(num_immigrants):
                    immigrant = toolbox.clone(best_individual)
                    toolbox.mutate(immigrant)  # Mutate immigrant rather than creating random individuals
                    population[random.randint(5, len(population) - 1)] = immigrant

                stagnation_counter = 0  # Reset stagnation counter after immigration

        # Get best individual after all generations
        best_individual = tools.selBest(population, 1)[0]
        best_distance = evaluate_individual(best_individual)[0]
        best_sub_route = split_route(best_individual)
        best_distances.append(best_distance)
        best_sub_routes.append(best_sub_route)

    return best_distances, best_sub_routes, distance_history



'''
# Run the genetic algorithm and print results
best_distances, best_sub_routes, distance_history = run_ga(runs=10, generations=100, pop_size=50)

# Print best routes
for i, sub_routes in enumerate(best_sub_routes):
    print(f"\nBest routes for run {i + 1}:")
    print_routes(sub_routes)

# Print statistics
print(f"Best distances over 10 runs: {best_distances}")
print(f"Average Best Distance: {mean(best_distances)}")
print(f"Standard Deviation: {stdev(best_distances)}")

# Plot convergence
plt.plot(distance_history)
plt.title('Convergence of Genetic Algorithm')
plt.xlabel('Generations')
plt.ylabel('Best Distance')
plt.grid(True)
plt.show()
'''
'''

# Run the genetic algorithm and print results
best_distances, best_sub_routes, distance_history = run_ga_with_immigration(runs=10, generations=100, pop_size=50)

# Print best routes
for i, sub_routes in enumerate(best_sub_routes):
    print(f"\nBest routes for run {i + 1}:")
    print_routes(sub_routes)

# Print statistics
print(f"Best distances over 10 runs: {best_distances}")
print(f"Average Best Distance: {mean(best_distances)}")
print(f"Standard Deviation: {stdev(best_distances)}")

# Plot convergence
plt.plot(distance_history)
plt.title('Convergence of Genetic Algorithm')
plt.xlabel('Generations')
plt.ylabel('Best Distance')
plt.grid(True)
plt.show()

'''

# Run the genetic algorithm and print results
best_distances, best_sub_routes, distance_history = run_ga_with_dynamic_mutation_and_refined_immigration(runs=10, generations=100, pop_size=50)

# Print best routes
for i, sub_routes in enumerate(best_sub_routes):
    print(f"\nBest routes for run {i + 1}:")
    print_routes(sub_routes)

# Print statistics
print(f"Best distances over 10 runs: {best_distances}")
print(f"Average Best Distance: {mean(best_distances)}")
print(f"Standard Deviation: {stdev(best_distances)}")

# Plot convergence
plt.plot(distance_history)
plt.title('Convergence of Genetic Algorithm')
plt.xlabel('Generations')
plt.ylabel('Best Distance')
plt.grid(True)
plt.show()