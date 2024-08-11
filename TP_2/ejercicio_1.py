import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from particle_swarm_optimization_algorithm import pso, OptimizationCriteria
from matplotlib import pyplot as plt

# Definir la función objetivo.
def objective_function(x):
    return 2 * np.sin(x) - (x ** 2) / 2

# Parámetros del problema.
num_particles = 2
num_iterations = 80
c1 = 2
c2 = 2
w = 0.7
inferior_limit = 0
superior_limit = 4

# Ejecución del PSO.
optimal_position, optimal_value, gbest_by_iteration = pso(objective_function,
                                                          num_dimensions=1,
                                                          num_particles=num_particles,
                                                          num_iterations=num_iterations,
                                                          c1=c1,
                                                          c2=c2,
                                                          w=w,
                                                          inferior_limit=inferior_limit,
                                                          superior_limit=superior_limit,
                                                          optimization_criteria=OptimizationCriteria.Maximize,
                                                          verbose=True)

# B. Solución óptima.
print(f'Solución óptima encontrada: x = {optimal_position[0]}')
print(f'Valor objetivo óptimo: f(x) = {optimal_value}')

# D. Graficar la función objetivo y el punto máximo encontrado.
x = np.linspace(inferior_limit, superior_limit, 500)
y = objective_function(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Funcion objetivo: $f(x) = 2\sin(x) - \\frac{x^2}{2}$')
plt.scatter(optimal_position, optimal_value, color='green', label='Optimal solution', zorder=5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Funcion objetivo y solucion optima')
plt.legend()
plt.grid(True)
plt.show()

# E. Graficar gbest en función de las iteraciones realizadas.
plt.figure(figsize=(10, 6))
plt.plot(range(num_iterations), gbest_by_iteration, color='blue', label='PSO con 2 particulas')
plt.xlabel('Iteracion')
plt.ylabel('Global Best (gbest)')
plt.title('Global Best (gbest) vs. Iteraciones')
plt.legend()
plt.grid(True)
plt.show()

# Inicializar la figura
plt.figure(figsize=(10, 6))

# Graficar gbest en función de las iteraciones realizadas para 2 partículas
plt.plot(range(num_iterations), gbest_by_iteration, color='blue', label='PSO con 2 partículas')

# Graficar gbest para diferentes números de partículas
particle_counts = [4, 10, 100, 200, 400]

for count in particle_counts:
    _, _, gbest_by_iteration = pso(objective_function,
                                   num_dimensions=1,
                                   num_particles=count,
                                   num_iterations=num_iterations,
                                   c1=c1,
                                   c2=c2,
                                   w=w,
                                   inferior_limit=inferior_limit,
                                   superior_limit=superior_limit,
                                   optimization_criteria=OptimizationCriteria.Maximize)
    plt.plot(range(num_iterations), gbest_by_iteration, label=f'{count} partículas')

# Etiquetas y título
plt.xlabel('Iteración')
plt.ylabel('Global Best (gbest)')
plt.title('Global Best (gbest) vs. Iteraciones para diferentes números de partículas')
plt.legend()
plt.grid(True)

# Mostrar el gráfico
plt.show()






