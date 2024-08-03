'''
Dada la siguiente función perteneciente a un paraboloide elíptico de la forma:
f(x, y) = (x - a)^2 + (y + b)^2
donde las constantes a y b son valores reales ingresados por el usuario a través de
la consola, con intervalos de:
−100 ≤ x ≤ 100 x ∈ ℝ
−100 ≤ y ≤ 100 y ∈ ℝ
−50 ≤ a ≤ 50 a ∈ ℝ
−50 ≤ b ≤ 50 b ∈ ℝ
escribir en Python un algoritmo PSO para la minimización de la función (1) que
cumpla con las siguientes consignas:

A. Transcribir el algoritmo utilizando los siguientes parámetros: número de
   partículas = 20, máximo número de iteraciones = 10, coeficientes de
   aceleración c1 = c2 = 2, peso de inercia w = 0.7. 
'''
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from particle_swarm_optimization_algorithm import pso, OptimizationCriteria, plot_gbests_by_iteration
from matplotlib import pyplot as plt

a = float(input('Ingresar valor del parámetro a'))
assert -50 <= a <= 50, 'a debe pertenecer al intervalo [-50, 50]'

b = float(input('Ingresar valor del parámetro b'))
assert -50 <= b <= 50, 'b debe pertenecer al intervalo [-50, 50]'

objective_function = lambda x, y: (x - a)**2 + (y + b)**2
num_dimensions = 2
num_particles = 20
num_iterations = 10
c1 = 2
c2 = 2
w = 0.7
inferior_limit = -100
superior_limit = 100
optimization_criteria = OptimizationCriteria.Minimize
verbose = True

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
C. Graficar usando matplotlib la función objetivo f(x, y) y agregar un punto rojo en
   donde el algoritmo haya encontrado el valor mínimo. El gráfico debe contener
   etiquetas en los ejes, leyenda y un título.
'''
x = np.linspace(inferior_limit, superior_limit, 100)
y = np.linspace(inferior_limit, superior_limit, 100)
x, y = np.meshgrid(x, y)
z = objective_function(x, y)

min_x = 0
min_y = 0
min_z = objective_function(gbest[0], gbest[1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', alpha=0.7)

ax.scatter(min_x, min_y, min_z, color='r', s=25)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y)')
plt.title(f'PSO sobre f(x, y) = (x - {a})^2 + (y + {b})^2')
plt.show()

'''
D. Realizar un gráfico de línea que muestre gbest en función de las iteraciones realizadas.
'''
plot_gbests_by_iteration(gbest_by_iteration, num_particles)

'''
E. Transcribir la solución óptima encontrada (dominio) y el valor objetivo óptimo (imagen).
'''
print(f'La solución óptima encontrada es ({gbest[0]}, {gbest[1]}) y su imagen es {objective_function(*gbest)}')

'''
F. Establecer el coeficiente de inercia w en 0, ejecutar el algoritmo y realizar
   observaciones/comentarios/conclusiones sobre los resultados observados.
'''
w = 0
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
Reducir el coeficiente de inercia w a 0 implica que la velocidad de cada partícula en la iteración
n no contribuye al determinar su velocidad en la iteración n+1. Esto hace que su movimiento sea más
errático, lo que les da más libertad para explorar el espacio de soluciones pero también hace que el
algoritmo sea más inestable y susceptible de no converger. También es posible que, sin w, el algoritmo
converja prematuramente en un óptimo local, ya que el movimiento de las partículas se halla determinado
por las mejores posiciones de cada iteración.
'''

'''
G. Reescribir el algoritmo PSO para que cumpla nuevamente con los ítems A hasta
   F pero usando la biblioteca pyswarm (from pyswarm import pso).
'''