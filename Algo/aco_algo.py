import pandas as pd
import numpy as np

# Load the distance matrix from Excel
distance_matrix = pd.read_excel("path_to_excel_file.xlsx", index_col=0)
distance_matrix = distance_matrix.values  # Convert to numpy array for easier indexing

# Define the depot (point_0 is depot) and the other points
depot = 0  # Depot is point_0

# Number of points (customers + depot)
num_points = distance_matrix.shape[0]

# Set up demand and service times
demand = np.ones(num_points)  # Every customer has a demand of 1
service_time = np.full(num_points, 10)  # Service time is 10 minutes for every point


def split_algorithm(customers, depot, vehicle_capacity, distance_matrix, service_time, max_trip_cost):
    n = len(customers)
    V = [float('inf')] * (n + 1)  # Initialize the cost for each customer
    V[0] = 0  # Starting cost at depot
    P = [-1] * (n + 1)  # To store split points

    for i in range(1, n + 1):
        load = 0
        cost = 0
        j = i
        while j <= n:
            load += 1  # Each customer has demand of 1, so we increment the load
            if j == i:
                # If it's the first customer in the trip, calculate cost from depot to this customer and back
                cost = distance_matrix[depot, customers[i - 1]] + service_time[customers[i - 1]] + distance_matrix[customers[i - 1], depot]
            else:
                # If it's a subsequent customer, calculate cost between customers and back to depot
                cost += distance_matrix[customers[j - 2], customers[j - 1]] + service_time[customers[j - 1]] + distance_matrix[customers[j - 1], depot]
            
            # Check if load and cost constraints are satisfied
            if load <= vehicle_capacity and cost <= max_trip_cost:
                if V[i - 1] + cost < V[j]:
                    V[j] = V[i - 1] + cost
                    P[j] = i - 1
            else:
                break  # If load or cost exceeds, stop the loop
            j += 1

    return V, P



def extract_vrp_solution(P, customers):
    n = len(customers)
    trips = []
    j = n
    while j > 0:
        i = P[j]
        trip = customers[i:j]  # Extract the trip between customers i and j
        trips.append(trip)
        j = i
    return trips






# Deap

from deap import base, creator, tools, algorithms

# Define the problem as a minimization problem
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Initialize the toolbox
toolbox = base.Toolbox()

# Register the genetic algorithm components
toolbox.register("indices", np.random.permutation, num_points - 1)  # Random sequence of customer points
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    # Individual is a sequence of customer indices (excluding depot)
    customers = individual
    
    # Run the split algorithm on the sequence of customers
    V, P = split_algorithm(customers, depot=depot, vehicle_capacity=10,  # Assume vehicle can hold 10 units
                           distance_matrix=distance_matrix, service_time=service_time, max_trip_cost=100)  # Assume max cost
    
    # Extract trips and calculate total cost
    trips = extract_vrp_solution(P, customers)
    total_cost = V[-1]  # Cost to reach the last customer
    return (total_cost,)

# Register evaluation and genetic operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic Algorithm parameters
population_size = 100
num_generations = 500
mutation_prob = 0.2
crossover_prob = 0.8

# Generate the initial population
population = toolbox.population(n=population_size)

# Run the Genetic Algorithm
algorithms.eaSimple(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob,
                    ngen=num_generations, verbose=True)































































































'''
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools

def load_distance_matrix(filename):
    # Load the Excel file
    df = pd.read_excel(filename, index_col=0)
    # Convert the DataFrame to a NumPy array
    distance_matrix = df.to_numpy()
    return distance_matrix

# Load the distance matrix from the Excel file
distance_matrix = load_distance_matrix('distance_matrix.xlsx')

# Initialize DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Ant", list, fitness=creator.FitnessMin)

# Define number of points and bins
num_points = distance_matrix.shape[0]  # Automatically get the number of points from the matrix
num_bins = 10
num_trucks = 2

# Pheromone matrix initialization
pheromone_matrix = np.ones((num_points, num_points))  # Initially, all pheromones are equal

# Alpha and Beta parameters for ACO
alpha = 1.0  # Pheromone importance
beta = 2.0   # Distance importance
evaporation_rate = 0.5  # Pheromone evaporation

# Function to calculate probability of selecting the next point
def calculate_transition_probabilities(current_point, visited, pheromone_matrix, distance_matrix):
    pheromones = np.copy(pheromone_matrix[current_point])
    distances = np.copy(distance_matrix[current_point])

    epsilon = 1e-6  # A small constant to avoid division by zero
    desirability = np.power(1 / (distances + epsilon), beta)

    # Pheromone and desirability weight
    probs = np.power(pheromones, alpha) * desirability

    # Mask visited nodes by setting their probability to 0
    for i in visited:
        probs[i] = 0

    # Normalize to get probabilities
    total = np.sum(probs)
    if total == 0:
        return np.ones(num_points) / num_points  # Uniform probabilities if all values are 0
    return probs / total

# Ant class: creates and evaluates a route
def generate_route(pheromone_matrix, distance_matrix):
    visited = [0]  # Start from depot
    route = [0]
    
    while len(visited) < num_points:
        current_point = route[-1]
        probabilities = calculate_transition_probabilities(current_point, visited, pheromone_matrix, distance_matrix)
        next_point = np.random.choice(range(num_points), p=probabilities)
        route.append(next_point)
        visited.append(next_point)

        # Simulate truck load limit and returning to depot
        if len(visited) % num_bins == 0:
            route.append(0)  # Return to depot

    return route

# Fitness function: minimizes total distance traveled
def evaluate_route(route, distance_matrix):
    total_distance = 0
    for i in range(1, len(route)):
        total_distance += distance_matrix[route[i-1]][route[i]]
    return (total_distance,)

# Pheromone update
def update_pheromones(pheromone_matrix, ant_population, distance_matrix):
    for ant in ant_population:
        for i in range(1, len(ant)):
            pheromone_matrix[ant[i-1]][ant[i]] += 1 / distance_matrix[ant[i-1]][ant[i]]
    # Evaporate pheromones
    pheromone_matrix *= (1 - evaporation_rate)



# DEAP integration
toolbox = base.Toolbox()

# Using lambda to pass pheromone_matrix and distance_matrix to the generate_route function
toolbox.register("ant", lambda: generate_route(pheromone_matrix, distance_matrix))
toolbox.register("population", tools.initRepeat, list, toolbox.ant)
toolbox.register("evaluate", evaluate_route, distance_matrix=distance_matrix)
toolbox.register("select", tools.selBest)

# Parameters
num_ants = 10
num_generations = 50

# Main ACO loop
population = toolbox.population(n=num_ants)
for gen in range(num_generations):
    # Evaluate population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Update pheromones
    update_pheromones(pheromone_matrix, population, distance_matrix)

    # Select the best ants
    best_ants = toolbox.select(population, k=5)
    print(f"Generation {gen}: Best distance = {min(fitnesses)}")

# Print best solution
best_solution = tools.selBest(population, k=1)[0]
print("Best Route:", best_solution)

'''







'''
import pandas as pd
import numpy as np
import random
from deap import base, creator, tools

def load_distance_matrix(filename):
    # Load the Excel file
    df = pd.read_excel(filename, index_col=0)
    # Convert the DataFrame to a NumPy array
    distance_matrix = df.to_numpy()
    return distance_matrix

# Load the distance matrix from the Excel file
distance_matrix = load_distance_matrix('distance_matrix.xlsx')

num_points = distance_matrix.shape[0]  # Number of locations (including depot)
num_bins = 10  # Maximum bins a truck can handle
num_trucks = 2

# Pheromone matrix initialization
pheromone_matrix = np.ones((num_points, num_points))

# Adjust pheromone evaporation, random exploration, and heuristic factor
pheromone_evaporation_rate = 0.8
pheromone_intensity = 2.0
random_exploration_prob = 0.3
beta = 1.5
alpha = 1.0

# Define the function to generate an ant
def generate_ant():
    """Generate an ant starting at the depot (point 0)."""
    start_point = 0
    return creator.Ant([start_point])

def update_ant(ant, pheromone):
    """Construct a solution (route) for an ant based on pheromone levels."""
    unvisited = set(range(1, num_points))  # Exclude the depot from unvisited
    current_truck_capacity = 0
    epsilon = 1e-6  # Small constant to avoid division by zero

    while unvisited:
        current_point = ant[-1]
        
        if current_truck_capacity == num_bins:  # Return to depot when truck is full
            ant.append(0)
            current_truck_capacity = 0
            continue

        if random.random() < random_exploration_prob:
            next_point = random.choice(list(unvisited))
        else:
            # Probabilistically choose next point based on pheromones and distances
            probabilities = []
            for next_point in unvisited:
                dist = distance_matrix[current_point][next_point] or epsilon
                pheromone_level = pheromone[current_point][next_point] ** alpha
                heuristic = (1.0 / dist) ** beta
                probabilities.append(pheromone_level * heuristic)

            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()  # Normalize

            next_point = np.random.choice(list(unvisited), p=probabilities)

        ant.append(next_point)
        unvisited.remove(next_point)
        current_truck_capacity += 1

    # Return to depot at the end
    ant.append(0)
    
    return ant

def evaluate_ant(ant):
    """Evaluate the total distance traveled by the trucks."""
    total_distance = 0
    for i in range(len(ant) - 1):
        total_distance += distance_matrix[ant[i]][ant[i + 1]]
    return total_distance,


# Elitism: Reinforce the best solution by keeping its pheromones strong
def reinforce_best(pheromone, best_ant):
    distance = evaluate_ant(best_ant)[0]
    for i in range(len(best_ant) - 1):
        pheromone[best_ant[i]][best_ant[i + 1]] += pheromone_intensity / distance


def update_pheromone(pheromone, population, best_ant):
    """Update the pheromone levels based on the routes taken by ants."""
    # Evaporate pheromones
    pheromone *= (1 - pheromone_evaporation_rate)

    # Reinforce pheromone levels based on ant solutions
    for ant in population:
        distance = evaluate_ant(ant)[0]
        for i in range(len(ant) - 1):
            pheromone[ant[i]][ant[i + 1]] += pheromone_intensity / distance

    # Ensure pheromone levels have a lower and upper bound
    pheromone = np.clip(pheromone, 0.01, 1.5)

    # Elitism: Reinforce the best ant's pheromone trail
    reinforce_best(pheromone, best_ant)

# DEAP setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Ant", list, fitness=creator.FitnessMin, best=None)
toolbox = base.Toolbox()
toolbox.register("ant", generate_ant)
toolbox.register("population", tools.initRepeat, list, toolbox.ant)
toolbox.register("update", update_ant)
toolbox.register("evaluate", evaluate_ant)
toolbox.register("update_pheromone", update_pheromone)

def main():
    pheromone = np.ones((num_points, num_points))
    pop = toolbox.population(n=20)

    best_ant = None  # Initialize best_ant

    for gen in range(100):  # Adjust generations as needed
        for ant in pop:
            ant[:] = toolbox.update(ant, pheromone)
            ant.fitness.values = toolbox.evaluate(ant)
        
        # Select the best ant from the population
        best_ant = tools.selBest(pop, 1)[0]
        
        # Update pheromones, passing the best_ant as argument
        toolbox.update_pheromone(pheromone, pop, best_ant)
        
        # Optional: Print best route of each generation
        print(f"Generation {gen}: Best distance = {best_ant.fitness.values[0]}")

if __name__ == "__main__":
    main()
'''