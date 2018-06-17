import matplotlib.pyplot as plt

import numpy
import random
import math

from pyeasyga import pyeasyga

# FUNCIONES
# Funcion sigmoide
def sigmoid(x):
    if x <0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-1 * x))

# def distance(x1,y1,x2,y2):
#     return

def normalize(angulo):
    a = angulo
    while (a > math.pi):
        a -= math.pi

    while (a < (-1 * math.pi)):
        a += math.pi

    return a



def create_individual(data):
    # individual que guardara todas las variables de la simulacion [tauSensor1, tauSensor2, tauMotorInner1, tauMotorInner2,|los de las intermedias|, biasSensor1, biasSensor2, biasMotorInner1, biasMotorInner2
    # ,| los de las intermedias|, gainSensor1, gainSensor2, gainMotorInner1, gainMotorInner2,| los de las intermedias |, LOS PESOS ]

    individual = []

    # Appendeamos las tau de los sensores (simetricas)
    tauSensores = round(random.uniform(0.1,1),2)
    individual.append(tauSensores)
    individual.append(tauSensores)

    # Appendeamos las tau de los motores (simetricas)
    tauMotores = round(random.uniform(0.1,1),2)
    individual.append(tauMotores)
    individual.append(tauMotores)

    # Appendeamos las tau de las neuronas intermedias
    for i in range(0, N):
        individual.append(round(random.uniform(0.1,1),2))

    assert(len(individual) == 4 + N)

    # Appendeamos los bias de los sensores (simetricos)
    biasSensores = round(random.uniform(-10,10),2)
    individual.append(biasSensores)
    individual.append(biasSensores)

    # Appendeamos los bias de los motores (simetricos)
    biasMotores = round(random.uniform(-10,10),2)
    individual.append(biasMotores)
    individual.append(biasMotores)

    # Appendeamos los bias de las neuronas intermedias
    for i in range(0, N):
        individual.append(round(random.uniform(-10,10),2))

    assert(len(individual) == (4 + N) * 2)

    # Appendeamos las ganancias de los sensores (simetricas)
    gainSensores = round(random.uniform(-10,10),2)
    individual.append(gainSensores)
    individual.append(gainSensores)

    # Appendeamos las ganancias de los motores (simetricas)
    gainMotores = round(random.uniform(-10,10),2)
    individual.append(gainMotores)
    individual.append(gainMotores)

    # Appendeamos las ganancias de las neuronas intermedias
    for i in range(0, N):
        individual.append(round(random.uniform(-10,10),2))

    assert(len(individual) == (4 + N) * 3)

    # Appendeamos los pesos entre neuronas intermedias y los sensores
    for i in range(0, N):
        individual.append(round(random.uniform(0,1),2)) # El peso de la neurona i con el sensor 1
        individual.append(round(random.uniform(0,1),2)) # El peso de la neurona i con el sensor 2

    # Appendeamos los pesos entre las neuronas motoras y cada una de las neuronas intermedias
    for i in range(0, N):
        individual.append(round(random.uniform(0,1),2)) # El peso del motor 1 con la neurona intermedia i

    # Appendeamos los pesos entre las neuronas motoras y cada una de las neuronas intermedias
    for i in range(0, N):
        individual.append(round(random.uniform(0,1),2)) # El peso del motor 2 con la neurona intermedia i

    assert(len(individual) == (4 + N) * 3 + (N * 2) *2)

    return individual

def crossover(parent_1, parent_2):
    crossover_index = random.randrange(1, len(parent_1))
    child_1 = parent_1[:crossover_index] + parent_2[crossover_index:]
    child_2 = parent_2[:crossover_index] + parent_1[crossover_index:]
    return child_1, child_2

def selection(population):
    return random.choice(population)

def mutate(individual):
    mutate_index = random.randrange(len(individual))
    if individual[mutate_index] == 0:
        individual[mutate_index] == 1
    else:
        individual[mutate_index] == 0

def fitness(individual,data):
    # Posicion del agente (aleatoria)
    xAgente = random.uniform(0,200)
    yAgente = random.uniform(0,200)

    anguloAgente = random.uniform(-1 * math.pi,math.pi)
    diametroAgente = 10

    # Posicion del sensor 1
    xSensor1 = xAgente + diametroAgente/2 * math.cos(anguloAgente + 0.349)
    ySensor1 = yAgente + diametroAgente/2 * math.sin(anguloAgente + 0.349)

    # Posicion del sensor 2
    xSensor2 = xAgente + diametroAgente/2 * math.cos(anguloAgente - 0.349)
    ySensor2 = yAgente + diametroAgente/2 * math.sin(anguloAgente - 0.349)

    # Velocidades del agente
    v = 0.0   # Velocidad lineal del agente
    w = 0.0   # Velocidad angular del agente

    # Otras variables
    outputSensor1 = 0.0
    outputSensor2 = 0.0
    outputInners = []
    outputMotorInner1 = 0.0
    outputMotorInner2 = 0.0

    for i in range(0, N):
        outputInners.append(0.0)

    # Ejecucion de la prueba
    times = 0

    while times < 1000:
        #print("%s %s %s %s", outputSensor1, outputSensor2, outputMotorInner1, outputMotorInner2)
# Actualizamos los sensores
        # Calcular distancia al objeto emisor de luz
        # Intensidad (salida) = maxIntensidad (100) / (distancia)²
        input1 = round((100 / math.pow(math.hypot(xSensor1 -xLuz, ySensor1 - yLuz),2)),4)

        input2 = round((100 / math.pow(math.hypot(xSensor2 -xLuz, ySensor2 - yLuz),2)),4)

        outputSensor1 = round((1 / individual[0]) * (-1 * outputSensor1 + input1),4)
        outputSensor2 = round((1 / individual[1]) * (-1 * outputSensor2 + input2),4)

# Actualizamos las neuronas de la capa intermedia
        for i in range(0, N):
            sumatorio = 0
            sumatorio += individual[(4 + N) * 3 + i] * sigmoid(individual[(4 + N) * 2] * (outputSensor1 + individual[4 + N]))
            sumatorio += individual[(4 + N) * 3 + i + 1] * sigmoid(individual[(4 + N + 1) * 2] * outputSensor2 + individual[4 + N + 1]))

            outputInners[i] = round((1 / individual[4 + i] * (-1 * outputInners[i] + sumatorio)),4)

# Actualizamos las neuronas de la capa de neuronas motoras
        # Para la neurona motora 1
        sumatorio = 0
        for i in range(0, N):
            sumatorio += individual[(4 + N) * 3 + (2 * N) + i] * sigmoid(individual[(4 + N) * 2 + 4 + i] * (outputInners[i] + individual[(4 + N) + 4 + i]))

        outputMotorInner1 = round((1 / individual[2] * (-1 * outputMotorInner1 + sumatorio)),4)

        # Para la neurona motora 2
        sumatorio = 0
        for i in range(0, N):
            sumatorio += individual[(4 + N) * 3 + (2 * N) + N + i] * sigmoid(individual[(4 + N) * 2 + 4 + i] * (outputInners[i] + individual[(4 + N) + 4 + i]))

        outputMotorInner2 = round((1 /individual[3] * (-1 * outputMotorInner2 + sumatorio)),4)

# Recalculamos la posicion del agente y de los sensores
        v = (outputMotorInner1 + outputMotorInner2) / 2
        w = (outputMotorInner1 - outputMotorInner2) / 2 * (diametroAgente / 2)

        # Las mapeamos entre -10 y 10 para que el agente no se vaya lejos
        v = v % 20.0 - 10.0
        w = w % 20.0 - 10.0

        # La posicion del agente
        anguloAgente = anguloAgente + w
        # Normalizamos el angulo
        anguloAgente = normalize(anguloAgente);
        xAgente += v * math.cos(anguloAgente)
        yAgente += v * math.sin(anguloAgente)

        # print("%s %s %s" ,xAgente, yAgente, anguloAgente)

        # Posicion del sensor 1
        xSensor1 = xAgente + diametroAgente/2 * math.cos(anguloAgente + 0.349)
        ySensor1 = yAgente + diametroAgente/2 * math.sin(anguloAgente + 0.349)

        # Posicion del sensor 2
        xSensor2 = xAgente + diametroAgente/2 * math.cos(anguloAgente - 0.349)
        ySensor2 = yAgente + diametroAgente/2 * math.sin(anguloAgente - 0.349)

        times+= 1
        # print("%s %s %s",xAgente, yAgente, anguloAgente)


    distanciaFinal = math.hypot(xAgente - xLuz, yAgente - yLuz)
    # print(distanciaFinal)
    if ( distanciaFinal == 0.0):
        return 1
    else: return 1 / distanciaFinal

# VARIABLES CONSTANTES Y GLOBALES
N = 5   # Numero de neuronas de la capa intermedia
xLuz = 160  # X de la luz en el tablero de 200 x 200
yLuz = 160  # Y de la luz en el tablero de 200 x 200

if __name__ == "__main__":

    data = [('xLuz', xLuz), ('yLuz', yLuz),('N', N)]
    ga = pyeasyga.GeneticAlgorithm(data,
                                    population_size=100,
                                    generations=10000,
                                    crossover_probability=0.8,
                                    mutation_probability=0.01,
                                    elitism=True,
                                    maximise_fitness=True)

    ga.create_individual = create_individual
    ga.crossover_function = crossover
    ga.mutate_function = mutate
    ga.selection_function = selection
    ga.fitness_function = fitness

    ga.run()
    print (ga.best_individual())

    # plt.plot(xLuz, yLuz, "ro")
    # plt.plot(historialX, historialY)
    # plt.show()
