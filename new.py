import random
import numpy as np
from deap import base, creator, tools
from statistics import mean, stdev

# Inputs
n = 7  # Number of cities (including depot)
D_max = 25  # Maximum allowed distance per truck route
P = 10  # Population size
M = 2  # Number of trucks
generations = 100  # Number of generations
runs = 30  # Number of times the algorithm will run

# Asymmetric distance matrix
distance_matrix = np.array([[99999, 2, 11, 10, 8, 7, 6],
                            [6, 99999, 9, 8, 4, 6, 6],
                            [3, 5, 99999, 12, 11, 8, 12],
                            [11, 9, 10, 99999, 9, 8, 11],
                            [5, 11, 11, 9, 99999, 11, 10],
                            [12, 8, 8, 5, 2, 99999, 11],
                            [7, 10, 9, 12, 10, 9, 99999]])

# DEAP base setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()

# Step 1: Create the initial population of a single long route
def create_individual():
    individual = [0] + random.sample(range(1, n), n - 1) + [0]  # Start and end at depot
    return individual

toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Step 2: Split a single long route into multiple sub-routes for multiple trucks
def split_route(individual):
    sub_routes = []
    current_route = [0]  # Each truck starts at the depot
    current_distance = 0

    for i in range(1, len(individual) - 1):
        next_city = individual[i]
        distance_to_next = distance_matrix[current_route[-1]][next_city]

        if current_distance + distance_to_next <= D_max:
            current_route.append(next_city)
            current_distance += distance_to_next
        else:
            current_route.append(0)  # End this truck's route at depot
            sub_routes.append(current_route)
            current_route = [0, next_city]
            current_distance = distance_matrix[0][next_city]

    current_route.append(0)
    sub_routes.append(current_route)

    return sub_routes

# Step 3: Fitness function (minimize total distance)
def route_distance(route):
    return sum(distance_matrix[route[i]][route[i+1]] for i in range(len(route) - 1))

def total_distance_of_sub_routes(sub_routes):
    return sum(route_distance(route) for route in sub_routes)

def evaluate_individual(individual):
    sub_routes = split_route(individual)
    return total_distance_of_sub_routes(sub_routes),

toolbox.register("evaluate", evaluate_individual)

# Step 4: Local Search Optimization using 2-Opt
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

# Step 5: Crossover (SCX - Sequential Constructive Crossover)
def scx_crossover(parent1, parent2):
    offspring = [0]
    current_city = 0

    for i in range(1, n):
        parent1_next_city = next_city_in_parent(parent1, current_city)
        parent2_next_city = next_city_in_parent(parent2, current_city)

        if distance_matrix[current_city][parent1_next_city] <= distance_matrix[current_city][parent2_next_city]:
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

# Step 6: Mutation (Swap Mutation)
def exchange_mutation(individual):
    idx1, idx2 = random.sample(range(1, len(individual) - 1), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# Step 7: Genetic Algorithm Loop
toolbox.register("select", tools.selRoulette)
toolbox.register("mate", scx_crossover)
toolbox.register("mutate", exchange_mutation)

# Running multiple simulations to collect statistics
best_distances = []
best_routes = []

for run in range(runs):
    population = toolbox.population(n=P)

    # Evaluate initial population
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    best_individual = None
    best_distance = float('inf')

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

        # Evaluate new offspring
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # Replace population
        population[:] = offspring

        # Track best individual of the generation
        current_best = tools.selBest(population, 1)[0]
        current_distance = evaluate_individual(current_best)[0]
        
        if current_distance < best_distance:
            best_distance = current_distance
            best_individual = current_best

    # Store best results for this run
    best_distances.append(best_distance)
    best_routes.append(split_route(best_individual))

# Calculate statistical analysis
avg_best_distance = mean(best_distances)
std_dev = stdev(best_distances)
min_best_distance = min(best_distances)
max_best_distance = max(best_distances)

# Output
print(f"Average Best Distance: {avg_best_distance}")
print(f"Standard Deviation: {std_dev}")
print(f"Minimum Best Distance: {min_best_distance}")
print(f"Maximum Best Distance: {max_best_distance}")


# Display the best route from one of the runs
best_run_index = best_distances.index(min_best_distance)
best_route_for_run = best_routes[best_run_index]
print(f"Best Route (from the best run): {best_route_for_run}")
