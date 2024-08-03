'''
Escribir un algoritmo PSO para la maximización de la función: y = sin(x) + sin(x^2)
En el intervalo de 0 ≤ x ≤ 10

A . Transcribir el algoritmo en Python con los siguientes parámetros: número de
partículas = 2, máximo número de iteraciones = 30, coeficientes de aceleración
c1 = c2 = 1.49, peso de inercia w = 0.5.
NOTA: Se define la función pso en el archivo particle_swarm_optimization_algorithm.py
      a los fines de tener una función reutilizable a parametrizarla según cada ejercicio.
'''

import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from particle_swarm_optimization_algorithm import pso

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

