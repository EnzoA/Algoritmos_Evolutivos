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
        restriction_functions=None,
        verbose=False):
    
    # Positions and velocities initialization.
    particles = _get_initial_particles(
        inferior_limit,
        superior_limit,
        num_particles,
        num_dimensions,
        restriction_functions)
    velocities = np.zeros((num_particles, num_dimensions))
    
    # Personal bests initialization.
    pbest = particles.copy()
    fitness_pbest = np.empty(num_particles)
    for i in range(num_particles):
        fitness_pbest[i] = objective_function(*[particles[i, j] for j in range(num_dimensions)])

    # Global best initialization.
    gbest_by_iteration = []
    gbest = (pbest[np.argmax(fitness_pbest)]
             if optimization_criteria == OptimizationCriteria.Maximize
             else pbest[np.argmin(fitness_pbest)])
    fitness_gbest = (np.max(fitness_pbest)
                     if optimization_criteria == OptimizationCriteria.Maximize
                     else np.min(fitness_pbest))

    # Search.
    for iteracion in range(num_iterations):
        for i in range(num_particles):
            # Evaluate the particle's fitness to get the new position.
            fitness = objective_function(*[particles[i, j] for j in range(num_dimensions)])
            is_new_pbest = _is_fittness_new_pbest(
                fitness,
                fitness_pbest[i],
                optimization_criteria,
                num_dimensions,
                particles[i],
                restriction_functions)

            if is_new_pbest:
                # Update the personal best.
                fitness_pbest[i] = fitness
                pbest[i] = particles[i].copy()

                # Update the global best.
                if (fitness > fitness_gbest
                    if optimization_criteria == OptimizationCriteria.Maximize
                    else fitness < fitness_gbest):
                    fitness_gbest = fitness
                    gbest = particles[i].copy()

            # Update the particle's velocity on each dimension.
            r1, r2 = np.random.rand(), np.random.rand()
            for d in range(num_dimensions):
                velocities[i][d] = (w * velocities[i][d] + c1 * r1 * (pbest[i][d] - particles[i][d]) + c2 * r2 * (gbest[d] - particles[i][d]))

            # Update the particle's position on each dimension.
            for d in range(num_dimensions):
                particles[i][d] = particles[i][d] + velocities[i][d]

                # If no restrictions are useed, keeping the particle inside the inferior and superior limits.
                if restriction_functions is None or len(restriction_functions) == 0:
                    particles[i][d] = np.clip(particles[i][d], inferior_limit, superior_limit)

            # Ensure that the new particle position satisfies the restrictions.
            if (restriction_functions is not None and
                len(restriction_functions) != 0 and
                not np.all([r(*[particles[i, j] for j in range(num_dimensions)]) for r in restriction_functions])):
                particles[i] = pbest[i].copy()

        if verbose:
            # Print each iteration's global best.
            print(f'Iteration number {iteracion + 1}: Global best position {gbest}, Value {fitness_gbest}')

        gbest_by_iteration.append(gbest)
    
    if verbose:
        print('\nOptimal position:', gbest)
        print('Optimal value:', fitness_gbest)

    # Return the global best, its fitness and the history of gbests by iteration.
    return gbest, fitness_gbest, gbest_by_iteration

def _get_initial_particles(inferior_limit, superior_limit, num_particles, num_dimensions, restriction_functions):
    if restriction_functions is None or len(restriction_functions) == 0:
        return np.random.uniform(inferior_limit, superior_limit, (num_particles, num_dimensions))
    else:
        particles = np.zeros((num_particles, num_dimensions))
        for i in np.arange(num_particles):
            # Iterate until the randomly generated particle satifies all the restrictions.
            while True:
                particles[i] = np.random.uniform(inferior_limit, superior_limit, num_dimensions)
                if np.all([r(*[particles[i, j] for j in range(num_dimensions)]) for r in restriction_functions]):
                    break
        return particles
    
def _is_fittness_new_pbest(fitness, fitness_pbest, optimization_criteria, num_dimensions, particle, restriction_functions):
    satisfies_restrictions = (restriction_functions is None or
                              len(restriction_functions) == 0 or
                              np.all([r(*[particle[i] for i in range(num_dimensions)]) for r in restriction_functions]))

    return (fitness > fitness_pbest and satisfies_restrictions
            if optimization_criteria == OptimizationCriteria.Maximize
            else fitness < fitness_pbest and satisfies_restrictions)

def plot_gbests_by_iteration(gbest_by_iteration, num_particles):
    plt.plot(np.arange(0, len(gbest_by_iteration)), gbest_by_iteration)
    plt.xlabel('Número de iteración')
    plt.ylabel('gbest')
    plt.title(f'gbest hallado en cada iteración con {num_particles} partículas')
    plt.legend([f'x{i + 1}' for i in range(len(gbest_by_iteration[0]))], loc='lower right')
    plt.show()