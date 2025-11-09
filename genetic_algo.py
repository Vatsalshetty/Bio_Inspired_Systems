import random
import math

# -----------------------------
# CONFIG
NUM_CITIES = 10
POP_SIZE = 100
GENERATIONS = 20
MUTATION_RATE = 0.02

# -----------------------------
# Generate random cities
cities = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(NUM_CITIES)]

# -----------------------------
# Distance between two cities
def distance(city1, city2):
    return math.hypot(city1[0] - city2[0], city1[1] - city2[1])

# Total path length (fitness is inverse)
def total_distance(tour):
    return sum(distance(cities[tour[i]], cities[tour[(i+1) % NUM_CITIES]]) for i in range(NUM_CITIES))

# -----------------------------
# Create random individual (a tour)
def create_individual():
    tour = list(range(NUM_CITIES))
    random.shuffle(tour)
    return tour

# Crossover: Order Crossover (OX)
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(NUM_CITIES), 2))
    child = [None] * NUM_CITIES

    # Copy slice from first parent
    child[start:end+1] = parent1[start:end+1]

    # Fill in the rest from second parent
    ptr = 0
    for city in parent2:
        if city not in child:
            while child[ptr] is not None:
                ptr += 1
            child[ptr] = city

    return child

# Mutation: Swap two cities
def mutate(tour):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(NUM_CITIES), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

# -----------------------------
# Initial population
population = [create_individual() for _ in range(POP_SIZE)]

# -----------------------------
# Main GA loop
for gen in range(GENERATIONS):
    # Evaluate fitness
    scored = [(ind, total_distance(ind)) for ind in population]
    scored.sort(key=lambda x: x[1])  # lower distance is better
    best = scored[0]

    print(f"Generation {gen+1}: Best distance = {best[1]:.2f}")

    # Selection: keep top 50%
    selected = [ind for ind, _ in scored[:POP_SIZE // 2]]

    # Reproduce
    children = []
    while len(children) < POP_SIZE:
        p1, p2 = random.sample(selected, 2)
        child = crossover(p1, p2)
        child = mutate(child)
        children.append(child)

    population = children

# Final best route
best_tour = scored[0][0]
print("\nBest tour found:", best_tour)


