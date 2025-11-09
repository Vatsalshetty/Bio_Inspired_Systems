import numpy as np

def de_jong(position):
    x, y = position
    return x**2 + y**2

num_particles = 30
dimensions = 2  
iterations = 10
w = 0.5
c1 = 1.5
c2 = 1.5
bounds = (-5.12, 5.12)

positions = np.random.uniform(bounds[0], bounds[1], (num_particles, dimensions))
velocities = np.random.uniform(-1, 1, (num_particles, dimensions))
pbest_positions = np.copy(positions)
pbest_scores = np.array([de_jong(p) for p in positions])

gbest_index = np.argmin(pbest_scores)
gbest_position = pbest_positions[gbest_index]
gbest_score = pbest_scores[gbest_index]

for t in range(iterations):
    for i in range(num_particles):
        r1 = np.random.rand(dimensions)
        r2 = np.random.rand(dimensions)

        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (pbest_positions[i] - positions[i]) +
                         c2 * r2 * (gbest_position - positions[i]))

        positions[i] += velocities[i]
        positions[i] = np.clip(positions[i], bounds[0], bounds[1])

        fitness = de_jong(positions[i])

        if fitness < pbest_scores[i]:
            pbest_positions[i] = positions[i]
            pbest_scores[i] = fitness

            if fitness < gbest_score:
                gbest_position = positions[i]
                gbest_score = fitness

    print(f"Iteration {t+1:3d} | Best Score: {gbest_score:.6f}")

print("\nBest Solution Found:")
print(f"Position: x = {gbest_position[0]:.6f}, y = {gbest_position[1]:.6f}")
print(f"Score: {gbest_score:.10f}")



