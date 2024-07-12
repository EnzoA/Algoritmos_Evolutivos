import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

class EvolutionaryAlgorithmBase(ABC):
    
    def __init__(self, population, fitness_thershold, selection_type):
        super().__init__()
        self._population = population
        self._fittest_solution = None
        self._fitness_threshold = fitness_thershold
        self._selection_type = selection_type

    def evolve(self, generations_number):
        for _ in np.arange(generations_number):
            self._fittest_solution = next(
                (item for item in self._population if self._get_fitness(item) > self._fitness_threshold),
                None
            )
            if self._fittest_solution is not None or self._population.size == 0:
                return self._fittest_solution
            parents = self._select_parents()


    @abstractmethod
    def _get_fitness(self, chromosome):
        pass

    def _select_parents(self):
        if self._selection_type == SelectionType.RWS:
            return self._select_parents_by_rws(self)
        elif self._selection_type == SelectionType.TS:
            return self._select_parents_by_ts
        elif self._selection_type == SelectionType.LRS:
            return self._select_parents_by_lrs()
        else:
            raise Exception('Unknown selection type ', self._selection_type)

    def _select_parents_by_rws(self):
        population_size = self._population.size
        total_fitness = 0
        selection_acum_probs = np.zeros((population_size))
        
        # Calculate the selection probability for each chromosome
        # as its fitness divided by the fitness of the entire population.
        # Store the cumulative probabilities.
        for chromosome in self._population:
            fitness = self._get_fitness(chromosome)
            np.append(selection_acum_probs, fitness)
            total_fitness += fitness
        selection_acum_probs /= total_fitness
        previous_acum_prob = 0
        for idx in np.arange(population_size):
            if idx != 0:
                selection_acum_probs[idx] += previous_acum_prob
            previous_acum_prob = selection_acum_probs[idx]
        
        # Generate as many uniformly distributed numbers as chromosomes the population has.
        # Make the selection.
        result = set()
        rands = np.random.uniform(size=population_size)
        for rand in rands:
            previous_acum_prob = 0
            for idx, acum_prob in enumerate(selection_acum_probs):
                if previous_acum_prob < rand < acum_prob:
                    result.add(self._population[idx])
                    break
                previous_acum_prob = acum_prob

        return np.array(list(result))

    def _select_parents_by_ts(self):
        pass

    def _select_parents_by_lrs(self):
        pass

class SelectionType(Enum):
    RWS = 1 # Roulette Wheel Selection aka Fitness Proportionate Selection (FPS)
    TS = 2  # Tournament Selection
    LRS = 3 # Linear Rank Selection
