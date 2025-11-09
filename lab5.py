import numpy as np
import random

values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
n_items = len(values)
n_nests = 15
Pa = 0.25
max_iter = 100

def fitness(x):
    total_weight = np.dot(x, weights)
    if total_weight > capacity:
        return 0 
    return np.dot(x, values)

def repair(x):
    while np.dot(x, weights) > capacity:
        idx = random.choice([i for i in range(n_items) if x[i]])
        x[idx] = 0
    return x

def levy_flight(Lambda=1.5):
    return np.random.normal(0, Lambda, n_items)

nests = [np.random.randint(0, 2, n_items) for _ in range(n_nests)]
nest_scores = [fitness(repair(x.copy())) for x in nests]

for generation in range(max_iter):
    for i in range(n_nests):
        new_nest = nests[i].copy()
        step = levy_flight()
        new_nest = np.abs(new_nest + step) > 0.5
        new_nest = new_nest.astype(int)
        new_nest = repair(new_nest)
        f_new = fitness(new_nest)
        if f_new > nest_scores[i]:
            nests[i] = new_nest
            nest_scores[i] = f_new
    
    indices = np.argsort(nest_scores)[:int(Pa * n_nests)]
    for idx in indices:
        nests[idx] = np.random.randint(0, 2, n_items)
        nests[idx] = repair(nests[idx])
        nest_scores[idx] = fitness(nests[idx])
best_idx = np.argmax(nest_scores)
print("Best solution:", nests[best_idx], "Value:", nest_scores[best_idx])
