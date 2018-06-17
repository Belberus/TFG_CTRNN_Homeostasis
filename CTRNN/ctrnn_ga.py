import matplotlib.pyplot as plt
from pyeasyga import pyeasyga
import math
import random

# FUNCIONES
# Funcion sigmoide
def sigmoid(x):
    if x < 0:
        return 1 - 1 / (1 + math.exp(x))
    return 1 / (1 + math.exp(-x))

def normalize(angulo):
    a = angulo
    while (a > math.pi):
        a -= math.pi

    while (a < (-1 * math.pi)):
        a += math.pi

    return a

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def create_individual(data):

    individual = []

    # Appendeamos la tau de los sensores
    tauSensores = round(random.uniform(0.1,1.0),2)
    individual.append(tauSensores)

    if (TAUSIMETRY):
        individual.append(tauSensores)
    else:
        tauSensores = round(random.uniform(0.1,1.0),2)
        individual.append(tauSensores)

    # Appendeamos la tau de los motores
    tauMotores = round(random.uniform(0.1,1.0),2)
    individual.append(tauMotores)

    if (TAUSIMETRY):
        individual.append(tauMotores)
    else:
        tauMotores = round(random.uniform(0.1,1.0),2)
        individual.append(tauMotores)

    # Appendeamos las tau de las neuronas intermedias
    for i in range(0, N):
        individual.append(round(random.uniform(0.1,1.0),2))

    assert(len(individual) == 4 + N)

    # Appendeamos el bias de los sensores
    biasSensores = round(random.uniform(-10,10),2)
    individual.append(biasSensores)

    if (BIASSIMETRY):
        individual.append(biasSensores)
    else:
        biasSensores = round(random.uniform(-10,10),2)
        individual.append(biasSensores)

    # Appendeamos el bias de los motores
    biasMotores = round(random.uniform(-10,10),2)
    individual.append(biasMotores)

    if (BIASSIMETRY):
        individual.append(biasMotores)
    else:
        biasMotores = round(random.uniform(-10,10),2)
        individual.append(biasMotores)

    # Appendeamos los bias de las neuronas intermedias
    for i in range(0, N):
        individual.append(round(random.uniform(-10,10),2))

    assert(len(individual) == (4 + N) * 2)

    # Appendeamos la ganancia de los sensores
    gainSensores = round(random.uniform(-10,10),2)
    individual.append(gainSensores)

    if (GAINSIMETRY):
        individual.append(gainSensores)
    else:
        gainSensores = round(random.uniform(-10,10),2)
        individual.append(gainSensores)

    # Appendeamos la ganancia de los motores
    gainMotores = round(random.uniform(-10,10),2)
    individual.append(gainMotores)

    if (GAINSIMETRY):
        individual.append(gainMotores)
    else:
        gainMotores = round(random.uniform(-10,10),2)
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

    # Velocidades del agente
    v = 0.0   # Velocidad lineal del agente
    w = 0.0   # Velocidad angular del agente

    # Otras variables
    outputSensor1 = 0.0
    outputSensor2 = 0.0
    outputInners = []
    outputMotor1 = 0.0
    outputMotor2 = 0.0

    for i in range(0, N):
        outputInners.append(0.0)

    # Ejecucion de la prueba
    times = 0

    while (times < ITERATIONS):
# Actualizamos los sensores
        input1 = 0.0
        input2 = 0.0

        # Calculamos el angulo entre la luz y el agente
        angAgenteLuz = normalize(math.atan2(yLuz - yAgente, xLuz - xAgente) - anguloAgente)

        # Limites de vision del sensor 1
        llimit1 = normalize(SEPARACIONSENSOR + VISIONSENSOR)   # 60º + 80º en radianes
        hlimit1 = normalize(SEPARACIONSENSOR - VISIONSENSOR)   # 60º - 80º en radianes

        # Limites de vision del sensor 2
        llimit2 = normalize(-SEPARACIONSENSOR + VISIONSENSOR)  # -60º + 80º en radianes
        hlimit2 = normalize(-SEPARACIONSENSOR - VISIONSENSOR)  # -60º - 80º en radianes

        # Comprobamos si los sensores estan activados
        s1Active = False
        s2Active = False

        if (angAgenteLuz <= llimit1):
            s1Active = True
            if (angAgenteLuz <= llimit2):
                s2Active = True
        elif (angAgenteLuz >= hlimit2):
            s2Active = True
            if (angAgenteLuz >= hlimit1):
                s1Active = True

        # Si estan activados calculamos el input
        if (s1Active == True):
            rad = normalize(anguloAgente + SEPARACIONSENSOR)
            xSensor1 = xAgente + (diametroAgente/2) * math.cos(rad)
            ySensor1 = yAgente + (diametroAgente/2) * math.sin(rad)

            # Distancia entre la luz y el sensor al cuadrado
            ds1 = math.hypot(xSensor1 - xLuz, ySensor1 - yLuz)**2

            # Distancia entre el centro del agente y la luz
            da = math.hypot(xAgente - xLuz, yAgente - yLuz)

            a = (((diametroAgente/2) * (diametroAgente/2)) + ds1) / (da * da)

            if (a <= 1.0):
                input1 = 100 / ds1

            # input1 = 100 /ds1

        if (s2Active == True):
            rad = normalize(anguloAgente - SEPARACIONSENSOR)
            xSensor2 = xAgente + (diametroAgente/2) * math.cos(rad)
            ySensor2 = yAgente + (diametroAgente/2) * math.sin(rad)

            # Distancia entre la luz y el sensor al cuadrado
            ds2 = math.hypot(xSensor2 - xLuz, ySensor1 - yLuz)**2

            # Distancia entre el centro del agente y la luz
            da = math.hypot(xAgente - xLuz, yAgente - yLuz)

            a = (((diametroAgente/2) * (diametroAgente/2)) + ds2) / (da * da)

            if (a <= 1.0):
                input2 = 100 / ds2

            # input2 = 100 /ds2


        outputSensor1 = (1 / individual[0]) * (-1 * outputSensor1 + input1)
        outputSensor2 = (1 / individual[1]) * (-1 * outputSensor2 + input2)

# Actualizamos las neuronas intermedias
        j = 0
        i = 0
        while (j < N):
            sumatorio = 0
            sumatorio += individual[(4 + N) * 3 + i] * sigmoid(individual[(4 + N) * 2] * (outputSensor1 + individual[2 + N]))
            sumatorio += individual[(4 + N) * 3 + i + 1] * sigmoid(individual[((4 + N) * 2 + 1)] * (outputSensor2 + individual[2 + N + 1]))
            outputInners[i] = (1 / individual[4 + i]) * (-1 * outputInners[i] + sumatorio)
            j += 2
            i += 1
        j = 0
        for i in range(0, N):
            sumatorio = 0
            sumatorio += individual[(4 + N) * 3 + j] * sigmoid(individual[(4 + N) * 2] * (outputSensor1 + individual[4 + N]))
            sumatorio += individual[(4 + N) * 3 + j + 1] * sigmoid(individual[((4 + N) * 2 + 1)] * (outputSensor2 + individual[4 + N + 1]))
            outputInners[i] = (1 / individual[4 + i]) * (-1 * outputInners[i] + sumatorio)
            j += 2

# Actualizamos las neuronas de la capa motora
        # Motor 1
        sumatorio = 0
        for i in range(0, N):
            sumatorio += individual[((4 + N) * 3) + (2 * N) + i] * sigmoid(individual[((4 + N) * 2) + 4 + i] * (outputInners[i] + individual[(4 + N) + 4 + i]))

        outputMotor1 = (1 / individual[2]) * (-1 * outputMotor1 + sumatorio)

        # Motor 2
        sumatorio = 0
        for i in range(0, N):
            sumatorio += individual[((4 + N) * 3) + (2 * N) + N + i] * sigmoid(individual[((4 + N) * 2) + 4 + i] * (outputInners[i] + individual[(4 + N) + 4 + i]))

        outputMotor2 = (1 / individual[3]) * (-1 * outputMotor2 + sumatorio)


# Recalculamos la posicion del agente y de los sensores
        # Recalculamos las velocidades
        v = (outputMotor1 + outputMotor2) / 2

        if (v > maxV):
            v = maxV
        elif (v < minV):
            v = minV

        w = ((outputMotor1 - outputMotor2) / 2) * (diametroAgente / 2)

        if (w > maxW):
            w = maxW
        elif (w < minW):
            w = minW


        # Recalculamos el angulo del agente
        anguloAgente = normalize(anguloAgente + w)

        # Recalculamos la posicion del agente
        xAgente += v * math.cos(anguloAgente)
        yAgente += v * math.sin(anguloAgente)

        times+= 1

    distanciaFinal = math.hypot(xAgente - xLuz, yAgente - yLuz)
    if ( distanciaFinal == 0.0):
        return 1
    elif ( math.isnan(distanciaFinal)):
        return 0
    else: return 1 / distanciaFinal


# VARIABLES CONSTANTES Y GLOBALES
N = 5   # Numero de neuronas de la capa intermedia
ITERATIONS = 600 # Numero de iteraciones de la prueba
SEPARACIONSENSOR = 1.0472 # Separacion de lo sensores respecto del eje del angulo del agente, 60º
VISIONSENSOR = 1.39626 # Amplitud angular en la que el sensor recoge lecturas, 80
xLuz = 160  # X de la luz en el tablero de 200 x 200
yLuz = 160  # Y de la luz en el tablero de 200 x 200
maxV = 15   # Maxima velocidad lineal permitida
minV = -15  # Minima velocidad lineal permitida
maxW = 15   # Maxima velocidad angular permitida
minW = -15  # Minima velocidad angular permitida

TAUSIMETRY = False
GAINSIMETRY = True
BIASSIMETRY = False

# Agente de diametro 10, con un sensor a +60º y -60º desde el punto a donde mira el agente, cada uno con un rango de deteccion de 80º

if __name__ == "__main__":

    data = [('xLuz', xLuz), ('yLuz', yLuz),('N', N)]
    ga = pyeasyga.GeneticAlgorithm(data,
                                    population_size=60,
                                    generations=5000,
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
    print("Best one: ")
    print (ga.best_individual())
    print("---------------------------------------------------------------------")
    for individual in ga.last_generation():
        print(individual)
