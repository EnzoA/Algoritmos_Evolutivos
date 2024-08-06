import numpy as np
from enum import Enum
from matplotlib import pyplot as plt

class OptimizationCriteria(Enum):
    Minimize = 1
    Maximize = 2

def pso(objective_function,
        num_dimensions,
        num_particles,
        num_iterations,
        c1,
        c2,
        w,
        inferior_limit,
        superior_limit,
        optimization_criteria,
        verbose=False):
    
    # Positions and velocities initialization.
    particles = np.random.uniform(inferior_limit, superior_limit, (num_particles, num_dimensions))
    velocities = np.zeros((num_particles, num_dimensions))
    
    # Personal bests initialization.
    pbest = particles.copy()
    fitness_pbest = np.empty(num_particles)
    for i in range(num_particles):
        fitness_pbest[i] = objective_function(*[particles[i, j] for j in range(num_dimensions)])

    # Global best initialization.
    gbest_by_iteration = []
    gbest = pbest[np.argmin(fitness_pbest)]
    fitness_gbest = np.min(fitness_pbest)

    # Search.
    for iteracion in range(num_iterations):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()

            # Update the particle's velocity on each dimension.
            for d in range(num_dimensions):
                velocities[i][d] = (w * velocities[i][d] + c1 * r1 * (pbest[i][d] - particles[i][d]) + c2 * r2 * (gbest[d] - particles[i][d]))

            # Update the particle's position on each dimension.
            for d in range(num_dimensions):
                particles[i][d] = particles[i][d] + velocities[i][d]

                # Keeping the particle inside the inferior and superior limits.
                particles[i][d] = np.clip(particles[i][d], inferior_limit, superior_limit)

            # Evaluate the particle's fitness to get the new position.
            fitness = objective_function(*[particles[i, j] for j in range(num_dimensions)])

            if (fitness > fitness_pbest[i]
                if optimization_criteria == OptimizationCriteria.Maximize
                else fitness < fitness_pbest[i]):
                # Update the personal best.
                fitness_pbest[i] = fitness
                pbest[i] = particles[i].copy()

                # Update the global best.
                if (fitness > fitness_gbest
                    if optimization_criteria == OptimizationCriteria.Maximize
                    else fitness < fitness_gbest):
                    fitness_gbest = fitness
                    gbest = particles[i].copy()

        if verbose:
            # Print each iteration's global best.
            print(f'Iteration number {iteracion + 1}: Global best position {gbest}, Value {fitness_gbest}')

        gbest_by_iteration.append(gbest)
    
    if verbose:
        print('\nOptimal position:', gbest)
        print('Optimal value:', fitness_gbest)

    # Return the global best, its fitness and the history of gbests by iteration.
    return gbest, fitness_gbest, gbest_by_iteration

def plot_gbests_by_iteration(gbest_by_iteration, num_particles):
    plt.plot(np.arange(0, len(gbest_by_iteration)), gbest_by_iteration)
    plt.xlabel('Número de iteración')
    plt.ylabel('gbest')
    plt.title(f'gbest hallado en cada iteración con {num_particles} partículas')
    plt.legend([f'x{i + 1}' for i in range(len(gbest_by_iteration[0]))], loc='lower right')
    plt.show()