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
from pyswarm import pso as pyswarm_pso

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
    optimization_criteria)

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
def plot_pso_result_3d(objective_function, gbest, inferior_limit, superior_limit, a, b):
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

plot_pso_result_3d(objective_function, gbest, inferior_limit, superior_limit, a, b)

'''
D. Realizar un gráfico de línea que muestre gbest en función de las iteraciones realizadas.
'''
plot_gbests_by_iteration(gbest_by_iteration, num_particles)

'''
E. Transcribir la solución óptima encontrada (dominio) y el valor objetivo óptimo (imagen).
'''
print(f'\nLa solución óptima encontrada es ({gbest[0]}, {gbest[1]}) y su imagen es {objective_function(*gbest)}. w={w}')

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
    optimization_criteria)
print(f'\nLa solución óptima encontrada es ({gbest[0]}, {gbest[1]}) y su imagen es {objective_function(*gbest)}. w={w}')

'''
Reducir el coeficiente de inercia w a 0 implica que la velocidad de cada partícula en la iteración
n no contribuye al determinar su velocidad en la iteración n+1. Esto hace que su movimiento sea más
errático, lo que les da más libertad para explorar el espacio de soluciones pero también hace que el
algoritmo sea más inestable y susceptible de no converger. También es posible que, sin w, el algoritmo
converja prematuramente en un óptimo local, ya que de esa forma el movimiento de las partículas se halla
determinado por las mejores posiciones de cada iteración. A pesar de esto, en sucesivas pruebas se
observaron resultados más cercanos al mínimo cuando se usó w=0.
'''

'''
G. Reescribir el algoritmo PSO para que cumpla nuevamente con los ítems A hasta
   F pero usando la biblioteca pyswarm (from pyswarm import pso).
'''
# Se redefine la misma función objetivo pero con la firma esperada por la librería pyswarm.
objective_function = lambda x: (x[0] - a)**2 + (x[1] + b)**2
w = 0.7
gbest, value = pyswarm_pso(
    func=objective_function,
    lb=[inferior_limit, inferior_limit],
    ub=[superior_limit, superior_limit],
    swarmsize=num_particles,
    omega=w,
    phip=c1,
    phig=c2,
    maxiter=num_iterations
)
plot_pso_result_3d(lambda x, y: (x - a)**2 + (y + b)**2, gbest, inferior_limit, superior_limit, a, b)
# NOTA: Respecto del punto "D. Realizar un gráfico de línea que muestre gbest en función de las iteraciones
# realizadas", pso de pyswarm no retorna el histórico de gbests para cada iteración.
print(f'\nLa solución óptima encontrada es ({gbest[0]}, {gbest[1]}) y su imagen es {objective_function(gbest)}. w={w}')

# Con w = 0
w = 0
gbest, value = pyswarm_pso(
    func=objective_function,
    lb=[inferior_limit, inferior_limit],
    ub=[superior_limit, superior_limit],
    swarmsize=num_particles,
    omega=w,
    phip=c1,
    phig=c2,
    maxiter=num_iterations
)
plot_pso_result_3d(lambda x, y: (x - a)**2 + (y + b)**2, gbest, inferior_limit, superior_limit, a, b)
print(f'\nLa solución óptima encontrada es ({gbest[0]}, {gbest[1]}) y su imagen es {objective_function(gbest)}. w={w}')

'''
En sucesivas pruebas no se observó una ventaja clara de alguna de las implementaciones de pso (la de pyswarm
y la propia). Ambas versiones del algoritmo obtuvieron resultados comparables ante la tarea de minimizar
f(x, y) = (x - a)^2 + (y + b)^2. Sí cabe destacar que con las dos variantes, eliminar el coeficiente
de inercia w al igualarlo a 0 brindó soluciones más cercanas al mínimo de la función.
'''