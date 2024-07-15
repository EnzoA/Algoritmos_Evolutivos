'''
Ejercicio 1: Crear en Python un vector columna A de 20 individuos binarios aleatorios de tipo
string. Crear un segundo vector columna B de 20 números aleatorios comprendidos
en el intervalo (0, 1). Mutar un alelo aleatorio a aquellos genes pertenecientes a los
cromosomas de A que tengan en su i-ésima fila un correspondiente de B inferior a
0.09. Almacenar los cromosomas mutados en un vector columna C y mostrarlos por
consola.
'''

import numpy as np

NUM_GENES = 5
POPULATION_SIZE = 20
MUTATION_THRESHOLD = 0.09

def mutate(chromosome, mutation_prob, mutation_threshold):
    if mutation_prob < mutation_threshold:
        idx_to_mutate = np.random.randint(chromosome.shape[0])
        chromosome[idx_to_mutate] = '0' if chromosome[idx_to_mutate] == '1' else '1'
        return chromosome
    else:
        return None

a = np.array(list(np.random.choice(['0', '1'], NUM_GENES) for _ in np.arange(POPULATION_SIZE)))
b = np.random.uniform(0.0, 1.0, POPULATION_SIZE)
c = {}
for i, (a_i, b_i) in enumerate(zip(a, b)):
    mutated = mutate(a_i, b_i, MUTATION_THRESHOLD)
    if mutated is not None:
        c[i] = mutated

print('Cromosomas A:', a)
print('\nVector aleatorio en intervalo (0, 1) B:', b)
print('\nCromosomas mutados C:')
for k, v in c.items():
    print(f'Cromosoma mutado en índice {k} de la población: {v}')