import numpy as np
from pyswarm import pso

# Definir las ecuaciones del sistema como funciones.
def equation1(x):
    x1, x2 = x
    return 3 * x1 + 2 * x2 - 9

def equation2(x):
    x1, x2 = x
    return x1 - 5 * x2 - 4

# Definir la función objetivo que minimiza la suma de los errores cuadrados.
def objective_function(x):
    return np.square(equation1(x)) + np.square(equation2(x))

# Definir los límites para x1 y x2.
lower_bounds = [-10, -10]
upper_bounds = [10, 10]

# Parámetros del PSO.
c1 = 2
c2 = 2
w = 0.5
num_particles = 30
max_iterations = 100

# Ejecutar el PSO.
optimal_solution, optimal_value = pso(objective_function,
                                      lower_bounds,
                                      upper_bounds,
                                      swarmsize=num_particles,
                                      maxiter=max_iterations,
                                      phip=c1,
                                      phig=c2,
                                      omega=w)

# Imprimir los resultados.
x1_opt, x2_opt = optimal_solution
print(f'Valores encontrados: x1 = {x1_opt}, x2 = {x2_opt}')
