import numpy as np
import random

class ACO_TSP:
    def __init__(self, distances, n_ants=10, n_iterations=100, alpha=1, beta=5, rho=0.5, Q=100):
        self.distances = distances
        self.n_cities = distances.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha      
        self.beta = beta       
        self.rho = rho         
        self.Q = Q            
        self.pheromone = np.ones((self.n_cities, self.n_cities))

    def run(self):
        best_length = float("inf")
        best_path = None

        for iteration in range(self.n_iterations):
            paths = []
            lengths = []

            for ant in range(self.n_ants):
                path = self.construct_solution()
                length = self.calculate_length(path)
                paths.append(path)
                lengths.append(length)

                if length < best_length:
                    best_length = length
                    best_path = path

            self.update_pheromones(paths, lengths)
            print(f"Iteration {iteration+1}: Best Length = {best_length}")

        return best_path, best_length

    def construct_solution(self):
        start = random.randint(0, self.n_cities - 1)
        path = [start]
        visited = {start}

        for _ in range(self.n_cities - 1):
            current = path[-1]
            next_city = self.choose_next_city(current, visited)
            path.append(next_city)
            visited.add(next_city)

        return path

    def choose_next_city(self, current, visited):
        probabilities = []
        for city in range(self.n_cities):
            if city not in visited:
                tau = self.pheromone[current][city] ** self.alpha
                eta = (1 / self.distances[current][city]) ** self.beta
                probabilities.append((city, tau * eta))
            else:
                probabilities.append((city, 0))

        total = sum(prob for _, prob in probabilities)
        r = random.random() * total
        cumulative = 0
        for city, prob in probabilities:
            cumulative += prob
            if cumulative >= r:
                return city

    def calculate_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.distances[path[i]][path[i+1]]
        length += self.distances[path[-1]][path[0]]
        return length

    def update_pheromones(self, paths, lengths):
        self.pheromone *= (1 - self.rho)   # evaporation
        for path, length in zip(paths, lengths):
            deposit = self.Q / length
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i+1]] += deposit
                self.pheromone[path[i+1]][path[i]] += deposit


if __name__ == "__main__":
    distances = np.array([
        [0, 2, 9, 10, 1],
        [2, 0, 6, 4, 3],
        [9, 6, 0, 8, 5],
        [10, 4, 8, 0, 7],
        [1, 3, 5, 7, 0]
    ])

    aco = ACO_TSP(distances, n_ants=10, n_iterations=10)
    best_path, best_length = aco.run()
    print("\nBest Path:", best_path)
    print("Best Length:", best_length)
