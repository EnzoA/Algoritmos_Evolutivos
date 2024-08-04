'''
Escribir un algoritmo PSO para la maximización de la función: y = sin(x) + sin(x^2)
En el intervalo de 0 ≤ x ≤ 10

A . Transcribir el algoritmo en Python con los siguientes parámetros: número de
    partículas = 2, máximo número de iteraciones = 30, coeficientes de aceleración
    c1 = c2 = 1.49, peso de inercia w = 0.5.
'''
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from particle_swarm_optimization_algorithm import pso, OptimizationCriteria, plot_gbests_by_iteration
from matplotlib import pyplot as plt

objective_function = lambda x: np.sin(x) + np.sin(x**2)
num_dimensions = 1
num_particles = 2
num_iterations = 30
c1 = 1.49
c2 = 1.49
w = 0.5
inferior_limit = 0
superior_limit = 10
optimization_criteria = OptimizationCriteria.Maximize
verbose = False

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
    verbose)

'''
B. Indicar la URL del repositorio en donde se encuentra el algoritmo PSO.
   
   Se define la función pso en el archivo particle_swarm_optimization_algorithm.py
   a los fines de tener una función reutilizable a parametrizarla según cada ejercicio.
'''

'''
C. Graficar usando matplotlib la función objetivo y agregar un punto negro en
   donde el algoritmo haya encontrado el valor máximo. El gráfico debe contener
   etiquetas en los ejes, leyenda y un título.
'''
def plot_pso_result(objective_function, gbest, num_particles):
    x = np.linspace(0, 10, 200)
    plt.figure(figsize=(8, 6))
    plt.plot(x, objective_function(x))
    gbest_value = objective_function(gbest)
    plt.scatter(gbest, (gbest_value), color='black', zorder=5)
    plt.text(
        gbest,
        gbest_value,
        f'Máximo hallado en ({gbest}, {gbest_value})',
        fontsize=12,
        ha='right')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'PSO sobre y = sin(x) + sin(x^2) con {num_particles} partículas')
    plt.grid(True)
    plt.show()

plot_pso_result(objective_function, gbest, num_particles)

'''
D. Realizar un gráfico de línea que muestre gbest en función de las iteraciones realizadas.
'''
plot_gbests_by_iteration(gbest_by_iteration, num_particles)

'''
E. Transcribir la solución óptima encontrada (dominio) y el valor objetivo óptimo (imagen). 
'''
print(f'La solución óptima encontrada es {gbest} y su imagen es {objective_function(gbest)}. Número de partículas: {num_particles}')

'''
F. Incrementar el número de partículas a 4, ejecutar la rutina, transcribir la
   solución óptima encontrada, transcribir el valor objetivo óptimo y realizar
   nuevamente los gráficos solicitados en C y D.
'''
num_particles = 4

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
    verbose)

print(f'\nLa solución óptima encontrada es {gbest} y su imagen es {objective_function(gbest)} Número de partículas: {num_particles}')

plot_pso_result(objective_function, gbest, num_particles)

plot_gbests_by_iteration(gbest_by_iteration, num_particles)

'''
G. Incrementar el número de partículas a 6, ejecutar la rutina, transcribir la
   solución óptima encontrada, transcribir el valor objetivo óptimo y realizar
   nuevamente los gráficos solicitados en C y D.
'''
num_particles = 6

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
    verbose)

print(f'\nLa solución óptima encontrada es {gbest} y su imagen es {objective_function(gbest)} Número de partículas: {num_particles}')

plot_pso_result(objective_function, gbest, num_particles)

plot_gbests_by_iteration(gbest_by_iteration, num_particles)

'''
H. Incrementar el número de partículas a 10, ejecutar la rutina, transcribir la
   solución óptima encontrada, transcribir el valor objetivo óptimo y realizar
   nuevamente los gráficos solicitados en C y D. 
'''
num_particles = 10

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
    verbose)

print(f'\nLa solución óptima encontrada es {gbest} y su imagen es {objective_function(gbest)} Número de partículas: {num_particles}')

plot_pso_result(objective_function, gbest, num_particles)

plot_gbests_by_iteration(gbest_by_iteration, num_particles)

'''
I. Realizar observaciones/comentarios/conclusiones sobre los resultados obtenidos.

En sucesivas ejecuciones del algoritmo empleando los diferentes números de partículas propuestos se
observó que, conforme el número de éstas crece, disminuyen las chances de que PSO se atasque en un
máximo local.
En el intervalo propuesto por el ejercicio, que es [0, 10], hay un máximo local cuyo valor imagen es
cercano al máximo global, aunque es menor a éste. En efecto, y(1.294) = 1.956, pero y(8.024) = 1.985.
Un mayor número de partículas permitiría una mejor exploración del espacio de búsqueda, disminuyendo
las probabilidades de atascarse en un óptimo local. Como desventaja, un número de partículas demasiado
alto repercute negativamente en el costo computacional.
'''