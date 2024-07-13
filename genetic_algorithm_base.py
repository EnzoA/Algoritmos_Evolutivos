import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

class GeneticAlgorithmBase(ABC):
    
    def __init__(
            self,
            population_size,
            chromosomes_size,
            selection_type,
            crossover_type,
            num_generations,
            crosssover_prob=0.9,
            mutation_prob=0.05,
            early_stop_fitness=None):
        
        super().__init__()

        assert population_size > 0, 'population_size must be greater than 0'
        assert chromosomes_size > 0, 'population_size must be greater than 0'
        assert selection_type is not None, 'selection_type is required'
        assert crossover_type is not None, 'crosover_type is required'
        assert crosssover_prob > 0, 'crosssover_prob must be greater than 0'
        assert mutation_prob > 0, 'mutation_prob must be greater than 0'

        self._population_size = population_size
        self._chromosomes_size = chromosomes_size
        self._selection_type = selection_type
        self._crossover_type = crossover_type
        self._num_generations = num_generations
        self._crossover_prob = crosssover_prob
        self._mutation_prob = mutation_prob
        self._early_stop_fitness = early_stop_fitness

    def evolve(self):
        population = np.random.randint(2, size=(self._population_size, self._chromosomes_size))

        for _ in np.arange(self._num_generations):
            if population.shape[0] == 0:
                return None

            # Early stop if configured.
            if self._early_stop_fitness is not None:
                fittest_solution = next(
                    (item for item in self._population_size if self._get_fitness(item) > self._fitness_threshold),
                    None
                )
                if fittest_solution is not None:
                    return fittest_solution

            # Selection.
            parents = self._select_parents(population)

            # Crossover.
            descendents = []
            for i in np.arange(parents.shape[0] - 1, step=2):
                descendent1, descendent2 = self._crossover(parents[i], parents[i + 1])
                descendents += [descendent1, descendent2]

            # Mutation.
            mutated_descendents = []
            for i, descendent in enumerate(descendents):
                mutated_descendents += [self._mutate(descendent)]
            population = np.array(mutated_descendents)

        # Return the fittest.
        idx = np.argmax(np.array([self._get_fitness(chromosome) for chromosome in population]))
        return population[idx]

    @abstractmethod
    def get_fenotype(self, chromosome):
        pass

    @abstractmethod
    def _get_fitness(self, chromosome):
        pass

    def _select_parents(self, population):
        if self._selection_type == SelectionType.RWS:
            return self._select_parents_by_rws(population)
        elif self._selection_type == SelectionType.TS:
            return self._select_parents_by_ts(population)
        elif self._selection_type == SelectionType.LRS:
            return self._select_parents_by_lrs(population)
        else:
            raise Exception('Unknown selection type ', self._selection_type)

    def _select_parents_by_rws(self, population):
        total_fitness = 0
        population_size = population.shape[0]
        selection_acum_probs = np.empty((population_size))
        
        # Calculate the selection probability for each chromosome
        # as its fitness divided by the fitness of the entire population.
        # Store the cumulative probabilities.
        for i, chromosome in enumerate(population):
            fitness = self._get_fitness(chromosome)
            selection_acum_probs[i] = fitness
            total_fitness += fitness
        selection_acum_probs /= total_fitness
        previous_acum_prob = 0
        for j in np.arange(population_size):
            if j != 0:
                selection_acum_probs[j] += previous_acum_prob
            previous_acum_prob = selection_acum_probs[j]
        
        # Generate as many uniformly distributed numbers as chromosomes the population has.
        # Make the selection.
        result = set()
        rands = np.random.uniform(size=population_size)
        for rand in rands:
            previous_acum_prob = 0
            for j, acum_prob in enumerate(selection_acum_probs):
                if previous_acum_prob < rand < acum_prob:
                    result.add(tuple(population[j]))
                    break
                previous_acum_prob = acum_prob

        return np.array(list(result), dtype='int')

    def _select_parents_by_ts(self, population):
        pass

    def _select_parents_by_lrs(self, population):
        pass

    def _crossover(self, chromosome1, chromosome2):
        if self._crossover_type == CrossoverType.SinglePoint:
            return self._crossover_single_point(chromosome1, chromosome2)
        elif self._crossover_type == CrossoverType.TwoPoint:
            pass
        elif self._crossover_type == CrossoverType.Uniform:
            pass
        else:
            raise Exception('Unknown crossover type ', self._crossover_type)
        
    def _crossover_single_point(self, chromosome1, chromosome2):
        if np.random.uniform() < self._crossover_prob:
            crossover_point = np.random.randint(1, chromosome1.size - 1)
            return (
                np.concat((chromosome1[:crossover_point], chromosome2[crossover_point:])),
                np.concat((chromosome2[:crossover_point], chromosome1[crossover_point:])))
        else:
            return chromosome1, chromosome2

    def _mutate(self, chromosome):
        for idx, allele in enumerate(chromosome):
            if np.random.random() < self._mutation_prob:
                chromosome[idx] = 0 if allele == 1 else 1
        return chromosome

class SelectionType(Enum):
    RWS = 1 # Roulette Wheel Selection aka Fitness Proportionate Selection (FPS)
    TS = 2  # Tournament Selection
    LRS = 3 # Linear Rank Selection

class CrossoverType(Enum):
    SinglePoint = 1
    TwoPoint = 2
    Uniform = 3

if __name__ == '__main__':
    
    class GeneticX2(GeneticAlgorithmBase):
        def __init__(self):
            super().__init__(
                population_size=4,
                chromosomes_size=5,
                selection_type=SelectionType.RWS,
                crossover_type=CrossoverType.SinglePoint,
                num_generations=20,
                mutation_prob=0.05)
            
        def get_fenotype(self, chromosome):
            return int(''.join(chromosome.astype(str)), 2)

        def _get_fitness(self, chromosome):
            return self.get_fenotype(chromosome) ** 2
            
    genetic_x2 = GeneticX2()
    solution = genetic_x2.evolve()
    print(solution)
    print(genetic_x2.get_fenotype(solution))