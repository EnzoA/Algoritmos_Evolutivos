import sys
import os
import random
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from genetic_algorithm_base import GeneticAlgorithmBase, SelectionType, CrossoverType
from matplotlib import pyplot as plt

def g(c):
    return 2*c / (4 + 0.8*c + c**2 + 0.2*c**3)

class GeneticGFunction(GeneticAlgorithmBase):

    def __init__(self):
        super().__init__()

    @property
    def _chromosomes_size(self):
        # Si se toma el intervalo [0, 10] y una precisión de 2 decimales, entonces
        # los cromosomas tendrán 10 bits: 2^9 < (10 - 0) * 100 = 1000 <= 2^10
        return 10

    def get_fenotype(self, chromosome):
        return round(self._chromosomes_size * int(''.join(chromosome.astype(str)), 2) / (2**self._chromosomes_size - 1), 2)
    
    def _get_fitness(self, chromosome):
        return g(self.get_fenotype(chromosome))

# a. Encontrar el valor aproximado de c para el cual g es máximo. Utilizar precisión de 2 decimales.
# b. Transcribir el algoritmo genético comentando brevemente las secciones de código que sean relevantes.
#    NOTA: El código se halla autodocumentado (y comentado en las secciones del código que lo ameritan) en la
#    definición de la clase base GeneticAlgorithmBase.
genetic_g_function = GeneticGFunction()
fittests_by_generation = genetic_g_function.evolve(
    population_size=100,
    selection_type=SelectionType.TS,
    crossover_type=CrossoverType.SinglePoint,
    num_generations=100,
    crosssover_prob=0.85,
    mutation_prob=0.07,
    verbose=True
)

def plot_genetic_run_results(fittest, fittests_by_generation):
    x = np.linspace(-1, 20, 400)
    y = g(x)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    if fittest is not None:
        solution_fenotype = genetic_g_function.get_fenotype(fittest)
        plt.scatter(solution_fenotype, g(solution_fenotype), color='red', zorder=5)
        plt.text(
            solution_fenotype,
            g(solution_fenotype),
            f'Solución hallada en ({solution_fenotype}, {g(solution_fenotype)})',
            fontsize=12,
            ha='right')
    if fittests_by_generation is not None:
        for f in random.sample(fittests_by_generation, k=15):
            fenotype = genetic_g_function.get_fenotype(f)
            plt.scatter(fenotype, g(fenotype), color='green', zorder=5)
    plt.xlabel('c')
    plt.ylabel('g')
    plt.title('Tasa de crecimiento g en función de la concentración c')
    plt.legend()

    plt.grid(True)
    plt.show()

# c. Graficar g en función de c en el intervalo [-1, 20] y agregar un punto rojo en la gráfica
#    en donde el algoritmo haya encontrado el valor máximo. El gráfico debe contener título,
#    leyenda y etiquetas en los ejes.
plot_genetic_run_results(fittest=fittests_by_generation[-1], fittests_by_generation=None)

# d. Graficar las mejores aptitudes encontradas en función de cada generación.
# El gráfico debe contener título, leyenda y etiquetas en los ejes.
# solution_fenotype = genetic_g_function.get_fenotype(fittests_by_generation[-1])
plot_genetic_run_results(fittest=None, fittests_by_generation=fittests_by_generation)