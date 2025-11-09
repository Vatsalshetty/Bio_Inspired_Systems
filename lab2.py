import numpy as np
import random
import operator

# Generate noisy data from a nonlinear function
def target_function(x):
    return 2 * x**3 - x + 1

def generate_data(n_points=30):
    xs = np.linspace(-2, 2, n_points)
    ys = np.array([target_function(x) for x in xs]) + np.random.normal(0, 1, n_points)
    return xs, ys

X_data, y_data = generate_data()

# Parameters
POP_SIZE = 50
GENE_LENGTH = 20
NUM_GENERATIONS = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7

# Function and terminal sets
FUNCTIONS = [('+', operator.add), ('-', operator.sub), ('*', operator.mul)]
TERMINALS = ['x', '1']
func_dict = {f[0]: f[1] for f in FUNCTIONS}

# Convert gene (list of symbols) into executable function
def express_gene(gene):
    stack = []
    for symbol in gene:
        if symbol in func_dict:
            try:
                b = stack.pop()
                a = stack.pop()
                func = func_dict[symbol]
                stack.append(lambda x, a=a, b=b, func=func: func(a(x), b(x)))
            except IndexError:
                stack.append(lambda x: 1)
        elif symbol == 'x':
            stack.append(lambda x: x)
        elif symbol == '1':
            stack.append(lambda x: 1)
        else:
            stack.append(lambda x: 1)
    return stack[-1] if stack else lambda x: 1

# Fitness: negative MSE
def fitness(gene):
    func = express_gene(gene)
    try:
        ys_pred = np.array([func(x) for x in X_data])
        mse = np.mean((ys_pred - y_data) ** 2)
        if np.isnan(mse) or np.isinf(mse):
            return -float('inf')
        return -mse
    except Exception:
        return -float('inf')

# Tournament selection
def selection(pop, k=3):
    selected = random.sample(pop, k)
    return max(selected, key=fitness)

# Crossover
def crossover(parent1, parent2):
    if len(parent1) != len(parent2):
        return parent1[:]
    point = random.randint(1, len(parent1) - 2)
    return parent1[:point] + parent2[point:]

# Mutation
def mutate(gene, mutation_rate=MUTATION_RATE):
    gene = gene[:]
    symbols = [f[0] for f in FUNCTIONS] + TERMINALS
    for i in range(len(gene)):
        if random.random() < mutation_rate:
            gene[i] = random.choice(symbols)
    return gene

# Initialize population
population = [[random.choice([f[0] for f in FUNCTIONS] + TERMINALS) for _ in range(GENE_LENGTH)] for _ in range(POP_SIZE)]

best_gene = None
best_fit = -float('inf')

# Main loop
for gen in range(NUM_GENERATIONS):
    new_population = []
    for _ in range(POP_SIZE):
        parent1 = selection(population)
        parent2 = selection(population)
        child = crossover(parent1, parent2) if random.random() < CROSSOVER_RATE else parent1[:]
        child = mutate(child)
        new_population.append(child)

    population = new_population

    generation_best = max(population, key=fitness)
    generation_best_fit = fitness(generation_best)

    if generation_best_fit > best_fit:
        best_fit = generation_best_fit
        best_gene = generation_best

    if gen % 10 == 0 or gen == NUM_GENERATIONS - 1:
        print(f"Generation {gen}: Best fitness = {best_fit:.6f}")

# Output best gene and prediction results
print("Best gene (symbolic expression):", best_gene)
best_func = express_gene(best_gene)
print("Sample predictions vs actual values:")
for x, y_true in zip(X_data[:10], y_data[:10]):
    try:
        y_pred = best_func(x)
        print(f"x={x:.3f}, Predicted={y_pred:.3f}, Actual={y_true:.3f}")
    except Exception:
        print(f"x={x:.3f}, Predicted=Error, Actual={y_true:.3f}")
