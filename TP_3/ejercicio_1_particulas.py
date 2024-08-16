import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from particle_swarm_optimization_algorithm import pso, OptimizationCriteria

# Definición de la función objetivo
objective_function = lambda a, b, c, d: 375.0 * a + 275.0 * b + 475.0 * c + 325.0 * d

# Parámetros fijos del algoritmo PSO
num_dimensions = 4
num_iterations = 50
c1 = 1.4944
c2 = 1.4944
w = 0.6
inferior_limit = 0
superior_limit = 200
optimization_criteria = OptimizationCriteria.Maximize

# Restricciones del problema
restriction_functions = [
    lambda a, b, c, d: a >= 0.0 and b >= 0.0 and c >= 0.0 and d >= 0.0,
    lambda a, b, c, d: 2.5 * a + 1.5 * b + 2.75 * c + 2.0 * d <= 640.0,
    lambda a, b, c, d: 3.5 * a + 3.0 * b + 3.0 * c + 2.0 * d <= 960.0
]

# Valores de partículas para probar
particle_numbers = [2, 5, 10, 20, 50, 100, 200]

# Almacenar los mejores gbest y sus valores
results = []

# Iterar sobre cada número de partículas
for num_particles in particle_numbers:
    gbest, value, gbest_by_iteration = pso(
        objective_function,
        num_dimensions,
        num_particles,
        num_iterations,
        c1,
        c2,
        w,
        inferior_limit,
        superior_limit,
        optimization_criteria,
        restriction_functions)

    results.append({
        'num_particles': num_particles,
        'gbest': gbest,
        'value': value
    })

    print(f'Número de partículas: {num_particles}, Mejor valor gbest: {value}, Solución: {gbest}')

# Encontrar el mejor resultado
best_result = max(results, key=lambda x: x['value'])

print(f'\nEl mejor resultado global se obtuvo con {best_result["num_particles"]} partículas:')
print(f'Solución: A={100 * int(best_result["gbest"][0])}, ' +
      f'B={100 * int(best_result["gbest"][1])}, ' +
      f'C={100 * int(best_result["gbest"][2])}, ' +
      f'D={100 * int(best_result["gbest"][3])}.')
print(f'Valor objetivo: {best_result["value"]}')
