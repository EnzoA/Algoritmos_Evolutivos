import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from genetic_algorithm_base import GeneticAlgorithmBase, SelectionType, CrossoverType

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
        c = self.get_fenotype(chromosome)
        return 2*c / (4 + 0.8*c + c**2 + 0.2*c**3)

genetic_g_function = GeneticGFunction()
genetic_g_function.evolve(
    population_size=200,
    selection_type=SelectionType.RWS,
    crossover_type=CrossoverType.SinglePoint,
    num_generations=3000,
    crosssover_prob=0.85,
    mutation_prob=0.07,
    verbose=True
)