import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from genetic_algorithm_base import GeneticAlgorithmBase, SelectionType, CrossoverType

class maxGenX2(GeneticAlgorithmBase):
    def __init__(self):
        super().__init__()
        
    @property
    def _chromosomes_size(self):
        return 5
            
    def get_fenotype(self, chromosome):
        return int(''.join(chromosome.astype(str)), 2)

    def _get_fitness(self, chromosome):
        return self.get_fenotype(chromosome) ** 2
            
genetic_x2 = maxGenX2()

genetic_x2.evolve(
        population_size=4,
        selection_type=SelectionType.RWS,
        crossover_type=CrossoverType.SinglePoint,
        num_generations=10,
        early_stop_fitness=999,
        verbose=True)
