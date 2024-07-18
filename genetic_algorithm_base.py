import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

class GeneticAlgorithmBase(ABC):
    
    def __init__(self):
        super().__init__()

    def evolve(
            self,
            population_size,
            selection_type,
            crossover_type,
            num_generations,
            crosssover_prob=0.9,
            mutation_prob=0.05,
            early_stop_fitness=None,
            verbose=False):

        assert self._chromosomes_size > 0, 'population_size must be greater than 0'
        assert population_size > 0, 'population_size must be greater than 0'
        assert selection_type is not None, 'selection_type is required'
        assert crossover_type is not None, 'crosover_type is required'
        assert crosssover_prob > 0, 'crosssover_prob must be greater than 0'
        assert mutation_prob > 0, 'mutation_prob must be greater than 0'

        population = np.random.randint(2, size=(population_size, self._chromosomes_size))

        for num_generation in np.arange(num_generations):
            self._log_output(f'===============\nGeneration: {num_generation + 1}', verbose)

            # Early stop if configured.
            if early_stop_fitness is not None:
                fittest_solution = next(
                    (item for item in population if self._get_fitness(item) > early_stop_fitness),
                    None
                )
                if fittest_solution is not None:
                    self._log_output(f'Early stopping. Fittest chromosome: {fittest_solution}. '
                                   + f'Fenotype: {self.get_fenotype(fittest_solution)}. '
                                   + f'Aptitude: {self._get_fitness(fittest_solution)}',
                                   verbose)
                    return fittest_solution

            # Selection.
            parents = self._select_parents(population, selection_type)

            # Crossover.
            descendents = []
            for i in np.arange(parents.shape[0] - 1, step=2):
                descendent1, descendent2 = self._crossover(parents[i], parents[i + 1], crossover_type, crosssover_prob)
                descendents.extend([descendent1, descendent2])

            # Mutation.
            mutated_descendents = []
            for i, descendent in enumerate(descendents):
                mutated_descendents.extend([self._mutate(descendent, mutation_prob)])
            population = np.array(mutated_descendents)

            # Get the fittest.
            idx = np.argmax(np.array([self._get_fitness(chromosome) for chromosome in population]))
            fittest_solution = population[idx]
            self._log_output(f'Fittest chromosome: {fittest_solution}. '
                       + f'Fenotype: {self.get_fenotype(fittest_solution)}. '
                       + f'Aptitude: {self._get_fitness(fittest_solution)}\n===============\n',
                       verbose)

        return fittest_solution

    @property
    @abstractmethod
    def _chromosomes_size(self):
        pass

    @abstractmethod
    def get_fenotype(self, chromosome):
        pass

    @abstractmethod
    def _get_fitness(self, chromosome):
        pass

    def _select_parents(self, population, selection_type):
        if selection_type == SelectionType.RWS:
            return self._select_parents_by_rws(population)
        elif selection_type == SelectionType.TS:
            return self._select_parents_by_ts(population)
        elif selection_type == SelectionType.LRS:
            return self._select_parents_by_lrs(population)
        else:
            raise Exception('Unknown selection type ', selection_type)

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
        result = []
        rands = np.random.uniform(size=population_size)
        for rand in rands:
            previous_acum_prob = 0
            for j, acum_prob in enumerate(selection_acum_probs):
                if previous_acum_prob < rand < acum_prob:
                    result.append(population[j])
                    break
                previous_acum_prob = acum_prob

        return np.array(result, dtype='int')

    def _select_parents_by_ts(self, population):
        population_size = population.shape[0]
        tournament_size = max(2, int(0.1 * population_size)) 
        population_indices = np.arange(population_size)
        result = []

        for _ in population:
            random_selection = population[np.random.choice(population_indices, tournament_size, replace=False)]
            best_index = 0
            for i in np.arange(random_selection.shape[0]):
                if self._get_fitness(random_selection[i]) > self._get_fitness(random_selection[best_index]):
                    best_index = i
            result.append(population[best_index])

        return np.array(result, dtype='int')

    def _select_parents_by_lrs(self, population):
        pass

    def _crossover(self, chromosome1, chromosome2, crossover_type, crossover_prob):
        if crossover_type == CrossoverType.SinglePoint:
            return self._crossover_single_point(chromosome1, chromosome2, crossover_prob)
        elif crossover_type == CrossoverType.TwoPoint:
            pass
        elif crossover_type == CrossoverType.Uniform:
            pass
        else:
            raise Exception('Unknown crossover type ', crossover_type)
        
    def _crossover_single_point(self, chromosome1, chromosome2, crossover_prob):
        if np.random.uniform() < crossover_prob:
            crossover_point = np.random.randint(1, chromosome1.size - 1)
            return (
                np.concat((chromosome1[:crossover_point], chromosome2[crossover_point:])),
                np.concat((chromosome2[:crossover_point], chromosome1[crossover_point:])))
        else:
            return chromosome1, chromosome2

    def _mutate(self, chromosome, mutation_prob):
        for idx, allele in enumerate(chromosome):
            if np.random.random() < mutation_prob:
                chromosome[idx] = 0 if allele == 1 else 1
        return chromosome

    def _log_output(self, message, verbose):
        if verbose == True:
            print(message)

class SelectionType(Enum):
    RWS = 1 # Roulette Wheel Selection aka Fitness Proportionate Selection (FPS)
    TS = 2  # Tournament Selection
    LRS = 3 # Linear Rank Selection

class CrossoverType(Enum):
    SinglePoint = 1
    TwoPoint = 2
    Uniform = 3

if __name__ == '__main__':
    
    # Use case example: Minimize x^2.
    class GeneticX2(GeneticAlgorithmBase):
        def __init__(self):
            super().__init__()
        
        @property
        def _chromosomes_size(self):
            return 5
            
        def get_fenotype(self, chromosome):
            return int(''.join(chromosome.astype(str)), 2)

        def _get_fitness(self, chromosome):
            return 1 / (self.get_fenotype(chromosome) ** 2 + 1)
            
    genetic_x2 = GeneticX2()

    solution = genetic_x2.evolve(
        population_size=4,
        selection_type=SelectionType.RWS,
        crossover_type=CrossoverType.SinglePoint,
        num_generations=30,
        early_stop_fitness=0.999,
        verbose=True)
