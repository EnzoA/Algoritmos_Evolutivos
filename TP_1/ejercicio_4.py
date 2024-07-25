import random
import numpy as np
import matplotlib.pyplot as plt

# Definir la función c(x, y)
def c(x, y):
    return 7.7 + 0.15 * x + 0.22 * y - 0.05 * x**2 - 0.016 * y**2 - 0.007 * x * y

# Parámetros
TAMANIO_POBLACION = 50
LONGITUD_CROMOSOMA = 30  # 15 bits para x y 15 bits para y
TASA_MUTACION = 0.01
TASA_CRUCE = 0.92
GENERACIONES = 100

# Función para decodificar un cromosoma
def decodificar(cromosoma):
    mitad = len(cromosoma) // 2
    x_bin = cromosoma[:mitad]
    y_bin = cromosoma[mitad:]
    x = -10 + int(x_bin, 2) * (20 / (2**mitad - 1))
    y = int(y_bin, 2) * (20 / (2**mitad - 1))
    return x, y

# Aptitud (c(x, y))
def aptitud(cromosoma):
    x, y = decodificar(cromosoma)
    return c(x, y)

# Inicializar la población
def inicializar_poblacion(tamanio_poblacion, longitud_cromosoma):
    poblacion = []
    for _ in range(tamanio_poblacion):
        cromosoma = ''.join([str(random.randint(0, 1)) for _ in range(longitud_cromosoma)])
        poblacion.append(cromosoma)
    return poblacion

# Selección por ruleta
def seleccion_ruleta(poblacion, aptitud_total):
    seleccion = random.uniform(0, aptitud_total)
    aptitud_actual = 0
    for individuo in poblacion:
        aptitud_actual += aptitud(individuo)
        if aptitud_actual > seleccion:
            return individuo

# Cruce monopunto
def cruce_mono_punto(progenitor1, progenitor2, tasa_cruce):
    if random.random() < tasa_cruce:
        punto_cruce = random.randint(1, len(progenitor1) - 1)
        descendiente1 = progenitor1[:punto_cruce] + progenitor2[punto_cruce:]
        descendiente2 = progenitor2[:punto_cruce] + progenitor1[punto_cruce:]
    else:
        descendiente1, descendiente2 = progenitor1, progenitor2
    return descendiente1, descendiente2

# Mutación
def mutacion(cromosoma, tasa_mutacion):
    cromosoma_mutado = ""
    for bit in cromosoma:
        if random.random() < tasa_mutacion:
            cromosoma_mutado += str(1 - int(bit))
        else:
            cromosoma_mutado += bit
    return cromosoma_mutado

# Algoritmo Genético
def algoritmo_genetico(tamaño_poblacion, longitud_cromosoma, tasa_mutacion, tasa_cruce, generaciones):
    poblacion = inicializar_poblacion(tamaño_poblacion, longitud_cromosoma)
    mejores_aptitudes = []

    for generacion in range(generaciones):
        # Calcular aptitud total
        aptitud_total = sum(aptitud(cromosoma) for cromosoma in poblacion)

        # Selección por ruleta
        progenitores = [seleccion_ruleta(poblacion, aptitud_total) for _ in range(tamaño_poblacion)]

        # Cruce
        descendientes = []
        for i in range(0, tamaño_poblacion, 2):
            descendiente1, descendiente2 = cruce_mono_punto(progenitores[i], progenitores[i + 1], tasa_cruce)
            descendientes.extend([descendiente1, descendiente2])

        # Mutación
        descendientes_mutados = [mutacion(descendiente, tasa_mutacion) for descendiente in descendientes]

        # Reemplazo de la población con elitismo
        poblacion.sort(key=aptitud)
        descendientes_mutados.sort(key=aptitud, reverse=True)
        for i in range(len(descendientes_mutados)):
            if aptitud(descendientes_mutados[i]) > aptitud(poblacion[i]):
                poblacion[i] = descendientes_mutados[i]

        # Mejor individuo de la generación
        mejor_individuo = max(poblacion, key=aptitud)
        mejores_aptitudes.append(aptitud(mejor_individuo))
        print(f"Generación {generacion + 1}: Mejor individuo = {decodificar(mejor_individuo)}, Aptitud = {aptitud(mejor_individuo)}")

    return max(poblacion, key=aptitud), mejores_aptitudes

# Ejecutar el algoritmo genético
mejor_solucion, mejores_aptitudes = algoritmo_genetico(TAMANIO_POBLACION, LONGITUD_CROMOSOMA, TASA_MUTACION, TASA_CRUCE, GENERACIONES)
mejor_x, mejor_y = decodificar(mejor_solucion)

# Mostrar la mejor solución
print(f"Mejor solución: x = {mejor_x:.3f}, y = {mejor_y:.3f}, Aptitud = {aptitud(mejor_solucion):.3f}")

# Gráficos
x_vals = np.linspace(-10, 10, 100)
y_vals = np.linspace(0, 20, 100)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
c_vals = c(x_grid, y_grid)

plt.figure(figsize=(10, 5))

# Gráfico de la función c(x, y)
plt.subplot(1, 2, 1)
plt.contourf(x_grid, y_grid, c_vals, cmap='viridis')
plt.colorbar(label='Concentración')
plt.scatter(mejor_x, mejor_y, color='red', label='Máximo encontrado')
plt.title('Distribución de la concentración de contaminante')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Gráfico de las mejores aptitudes por generación
plt.subplot(1, 2, 2)
plt.plot(range(1, GENERACIONES + 1), mejores_aptitudes, marker='o', label='Mejor aptitud')
plt.title('Mejor aptitud por generación')
plt.xlabel('Generación')
plt.ylabel('Mejor aptitud')
plt.legend()

plt.tight_layout()
plt.show()
