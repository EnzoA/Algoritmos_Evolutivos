'''
Escribir un algoritmo PSO para la maximización de la función: y = sin(x) + sin(x^2)
En el intervalo de 0 ≤ x ≤ 10

A . Transcribir el algoritmo en Python con los siguientes parámetros: número de
    partículas = 2, máximo número de iteraciones = 30, coeficientes de aceleración
    c1 = c2 = 1.49, peso de inercia w = 0.5.
'''

import sys
import os
import random
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from particle_swarm_optimization_algorithm import pso
from matplotlib import pyplot as plt

gbest, value = pso(objective_function=lambda x: np.sin(x) + np.sin(x**2),
                   num_dimensions=1,
                   num_particles=20,
                   num_iterations=10,
                   c1=1.49,
                   c2=1.49,
                   w=0.5,
                   inferior_limit=0,
                   superior_limit=10,
                   verbose=True)

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
def plot_pso_run_results(gbest, gbests_by_iteration):
    x = np.linspace(0, 10, 200)
    y = lambda x: np.sin(x) + np.sin(x**2)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y(x))
    if gbest is not None:
        gbest_value = y(gbest)
        plt.scatter(gbest, (gbest_value), color='black', zorder=5)
        plt.text(
            gbest,
            gbest_value,
            f'Máximo hallado en ({gbest}, {gbest_value})',
            fontsize=12,
            ha='right')
    if gbests_by_iteration is not None:
        for g in random.sample(gbests_by_iteration, k=15):
            gbest_value = y(g)
            plt.scatter(gbest, gbest_value, color='green', zorder=5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y = sin(x) + sin(x^2)')
    plt.legend()

    plt.grid(True)
    plt.show()

plot_pso_run_results(gbest=gbest, gbests_by_iteration=None)

'''
D. Realizar un gráfico de línea que muestre gbest en función de las iteraciones realizadas.
'''
