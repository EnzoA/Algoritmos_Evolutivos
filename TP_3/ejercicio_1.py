'''
Una fábrica produce cuatro tipos de partes automotrices. Cada una de ellas primero
se fabrica y luego se le dan los acabados. Las horas de trabajador requeridas y la
utilidad para cada parte son las siguientes:

Tiempo de fabricación hr/100 unidades: A: 2.5; B: 1.5; C: 2.75; D: 2
Tiempo de acabados hr/100 unidades: A: 3.5; B: 3; C: 3; D: 2
Utilidad $/100 unidades: A: 375; B: 275; C: 475; D: 325

Las capacidades de los talleres de fabricación y acabados para el mes siguiente son
de 640 y 960 horas, respectivamente. Determinar mediante un algoritmo PSO con
restricciones (sin usar bibliotecas para PSO) que cantidad de cada parte debe
producirse a fin de maximizar la utilidad y resolver las siguientes consignas: 

A. Transcribir el algoritmo escrito en Python a un archivo .pdf de acuerdo a los
   siguientes parámetros: número de partículas = 20, máximo número de
   iteraciones 50, coeficientes de aceleración c1 = c2 = 1.4944, factor de inercia
   w = 0.6.
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from particle_swarm_optimization_algorithm import pso, OptimizationCriteria, plot_gbests_by_iteration

objective_function = lambda a, b, c, d: 375.0 * a + 275.0 * b + 475.0 * c + 325.0 * d
num_dimensions = 4
num_particles = 20
num_iterations = 50
c1 = 1.4944
c2 = 1.4944
w = 0.6
inferior_limit = 0
superior_limit = 200
optimization_criteria = OptimizationCriteria.Maximize
restriction_functions = [
    lambda a, b, c, d: a >= 0.0 and b >= 0.0 and c >= 0.0 and d >= 0.0,
    lambda a, b, c, d: 2.5 * a + 1.5 * b + 2.75 * c + 2.0 * d <= 640.0,
    lambda a, b, c, d: 3.5 * a + 3.0 * b + 3.0 * c + 2.0 * d <= 960.0
]

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

'''
B. Transcribir al .pdf la solución óptima encontrada (dominio) y el valor objetivo óptimo (imagen).
'''
print(f'\nLa solución óptima encontrada es ' +
      f'A={100 * int(gbest[0])}, ' +
      f'B={100 * int(gbest[1])}, ' +
      f'C={100 * int(gbest[2])} y ' +
      f'D={100 * int(gbest[3])}. ' +
      f'Su imagen es {objective_function(*[int(x) for x in gbest])}.')

'''
C. Indicar en el .pdf la URL del repositorio en donde se encuentra el algoritmo PSO.

   Se define la función pso en el archivo particle_swarm_optimization_algorithm.py
   a los fines de tener una función reutilizable a parametrizarla según cada ejercicio.
'''

'''
D. Realizar un gráfico de línea que muestre gbest (eje de ordenadas) en función de
   las iteraciones realizadas (eje de abscisas). El gráfico debe contener etiquetas en
   los ejes, leyenda y un título. El gráfico debe ser pegado en el .pdf.
'''
plot_gbests_by_iteration(gbest_by_iteration, num_particles)

'''
E. Explicar (en el .pdf) y demostrar (desde el código fuente) que sucede si se reduce en 1 unidad el tiempo de acabado
de la parte B.
'''

restriction_functions = [
    lambda a, b, c, d: a >= 0.0 and b >= 0.0 and c >= 0.0 and d >= 0.0,
    lambda a, b, c, d: 2.5 * a + 1.5 * b + 2.75 * c + 2.0 * d <= 640.0,
    lambda a, b, c, d: 3.5 * a + 2.0 * b + 3.0 * c + 2.0 * d <= 960.0
]

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

print(f'\nLa solución óptima encontrada es ' +
      f'A={100 * int(gbest[0])}, ' +
      f'B={100 * int(gbest[1])}, ' +
      f'C={100 * int(gbest[2])} y ' +
      f'D={100 * int(gbest[3])}. ' +
      f'Su imagen es {objective_function(*[int(x) for x in gbest])}.')

plot_gbests_by_iteration(gbest_by_iteration, num_particles)


'''
F. Realizar observaciones/comentarios/conclusiones en el .pdf acerca de qué cantidad mínima de partículas es reco-
mendable utilizar para este problema específicamente.
'''

restriction_functions = [
    lambda a, b, c, d: a >= 0.0 and b >= 0.0 and c >= 0.0 and d >= 0.0,
    lambda a, b, c, d: 2.5 * a + 1.5 * b + 2.75 * c + 2.0 * d <= 640.0,
    lambda a, b, c, d: 3.5 * a + 3.0 * b + 3.0 * c + 2.0 * d <= 960.0
]

particle_numbers = [2, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200]

results = []

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


