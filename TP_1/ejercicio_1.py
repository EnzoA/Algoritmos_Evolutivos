'''
Ejercicio 1: Crear en Python un vector columna A de 20 individuos binarios aleatorios de tipo
string. Crear un segundo vector columna B de 20 números aleatorios comprendidos
en el intervalo (0, 1). Mutar un alelo aleatorio a aquellos genes pertenecientes a los
cromosomas de A que tengan en su i-ésima fila un correspondiente de B inferior a
0.09. Almacenar los cromosomas mutados en un vector columna C y mostrarlos por
consola.
'''

import numpy as np

size = 20
threshold = 0.09

a = np.random.choice(['0', '1'], size)
b = np.random.uniform(0.0, 1.0, size)
c = np.array(list(
    ('0'
     if x == '1' and y < threshold
     else ('1'
           if x == '0' and y < threshold
           else x)
    for x, y in zip(a, b))
))

print('Vector binario string A: ', a)
print('\nVector aleatorio en intervalo (0, 1) B:', b)
print('\nVector C. Su i-ésima fila es el alelo correspondiente de A mutado ' +
      'si su correspondiente en B es menor a 0.09 o dicho alelo sin mutar en caso contrario:', c)