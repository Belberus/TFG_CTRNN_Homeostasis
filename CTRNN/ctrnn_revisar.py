import matplotlib.pyplot as plt
from pyeasyga import pyeasyga

import random
import math

# FIJADO A 2 NEURONAS SENSORAS Y DOS NEURONAS MOTORAS, SIN SIMETRIA
def sigmoid(x):
    valor = 0.0
    if x < 0:
        valor = 1 - 1 / (1 + math.exp(x))
    else:
        valor = 1 / (1 + math.exp(-x))

    # print(valor)
    return valor
def normalize(angulo):
    a = angulo
    while (a > math.pi):
        a -= math.pi

    while (a < (-1 * math.pi)):
        a += math.pi

    return a

def distance(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def crossover(parent_1, parent_2):
    crossover_index = random.randrange(1, len(parent_1))
    child_1 = parent_1[:crossover_index] + parent_2[crossover_index:]
    child_2 = parent_2[:crossover_index] + parent_1[crossover_index:]
    return child_1, child_2

def selection(population):
    return random.choice(population)

def mutate(individual):
    mutate_index = random.randrange(len(individual))
    if ((mutate_index >= 0) and (mutate_index <= 3)):
        individual[mutate_index] = round(random.uniform(0.1,1.0),2)
    else:
        individual[mutate_index] = round(random.uniform(-10,10),2)


def create_individual(data):
    individual = []

    # Appendeamos las tau
    individual.append(round(random.uniform(0.4,4.0),2)) # individual[0] tau sensor 1
    individual.append(round(random.uniform(0.4,4.0),2)) # individual[1] tau sensor 2
    individual.append(round(random.uniform(0.4,4.0),2)) # individual[2] tau motor 1
    individual.append(round(random.uniform(0.4,4.0),2)) # individual[3] tau motor 2

    # Appendeamos las ganancias
    gainSensor = round(random.uniform(0.1,10),2)
    individual.append(gainSensor)  # individual[4] ganancia sensor 1
    individual.append(gainSensor)  # individual[5] ganancia sensor 2

    gainMotor = round(random.uniform(0.1,10),2)
    individual.append(gainMotor)  # individual[6] ganancia motor 1
    individual.append(gainMotor)  # individual[7] ganancia motor 2

    # Appendeamos los bias
    individual.append(round(random.uniform(-3,3),2))  # individual[8] bias sensor 1
    individual.append(round(random.uniform(-3,3),2))  # individual[9] bias sensor 2
    individual.append(round(random.uniform(-3,3),2))  # individual[10] bias motor 1
    individual.append(round(random.uniform(-3,3),2))  # individual[11] bias motor 2

    # Appendeamos los pesos
    individual.append(round(random.uniform(-8,8),2))     # individual[12] peso sensor1 al sensor2
    individual.append(round(random.uniform(-8,8),2))     # individual[13] peso sensor1 al motor1
    individual.append(round(random.uniform(-8,8),2))     # individual[14] peso sensor1 al motor2
    individual.append(round(random.uniform(-8,8),2))     # individual[15] peso sensor2 al sensor1
    individual.append(round(random.uniform(-8,8),2))     # individual[16] peso sensor2 al motor1
    individual.append(round(random.uniform(-8,8),2))     # individual[17] peso sensor2 al motor2
    individual.append(round(random.uniform(-8,8),2))     # individual[18] peso motor1 al motor2
    individual.append(round(random.uniform(-8,8),2))     # individual[19] peso motor1 al sensor1
    individual.append(round(random.uniform(-8,8),2))     # individual[20] peso motor1 al sensor2
    individual.append(round(random.uniform(-8,8),2))     # individual[21] peso motor2 al motor1
    individual.append(round(random.uniform(-8,8),2))     # individual[22] peso motor2 al sensor1
    individual.append(round(random.uniform(-8,8),2))     # individual[23] peso motor2 al sensor2

    return individual

historialX = []
historialY = []
def fitness(individual,data):
    # Asignamos una posicion aleatoria al agente, dentro del tablero 200x200 y a una distancia de DISTANCIA unidades
    angulo = random.random() * (2 * math.pi)
    xAgente = math.cos(angulo) * DISTANCIA
    yAgente = math.sin(angulo) * DISTANCIA

    anguloAgente = random.random() * (2 * math.pi)
    diametroAgente = 1

    # Salidas
    outputSensor1 = 0.0
    outputSensor2 = 0.0
    outputMotor1 = 0.0
    outputMotor2 = 0.0

    ciclos = 0

    while (ciclos < ITERACIONES):
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

# ACTUALIZAMOS LOS SENSORES
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
            rad = normalize(anguloAgente - SEPARACIONSENSOR)
            xSensor1 = xAgente + (diametroAgente/2) * math.cos(rad)
            ySensor1 = yAgente + (diametroAgente/2) * math.sin(rad)

            #print(xSensor1, " ", ySensor1, "  /  ", xAgente, " ", yAgente)
            # Distancia entre la luz y el sensor al cuadrado
            # ds1 = math.hypot(xSensor1 - xLuz, ySensor1 - yLuz)**2
            ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2

            # Distancia entre el centro del agente y la luz
            # da = math.hypot(xAgente - xLuz, yAgente - yLuz)
            da = distance(xAgente, yAgente, xLuz, yLuz)

            a = (((diametroAgente/2) * (diametroAgente/2)) + ds1) / (da * da)

            if (a <= 1.0):
                input1 = 100 / ds1

        if (s2Active == True):
            rad = normalize(anguloAgente - SEPARACIONSENSOR)
            xSensor2 = xAgente + (diametroAgente/2) * math.cos(rad)
            ySensor2 = yAgente + (diametroAgente/2) * math.sin(rad)

            # Distancia entre la luz y el sensor al cuadrado
            # ds2 = math.hypot(xSensor2 - xLuz, ySensor1 - yLuz)**2
            ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2

            # Distancia entre el centro del agente y la luz
            # da = math.hypot(xAgente - xLuz, yAgente - yLuz)
            da = distance(xAgente, yAgente, xLuz, yLuz)

            a = (((diametroAgente/2) * (diametroAgente/2)) + ds2) / (da * da)

            if (a <= 1.0):
                input2 = 100 / ds2

        # print(input1, " ", input2)

        # Sensor 1
        sumatorio = 0.0
        sumatorio += individual[15] * sigmoid(individual[5] * (outputSensor2 + individual[9]))  # Sensor 2
        sumatorio += individual[19] * sigmoid(individual[6] * (outputMotor1 + individual[10]))  # Motor 1
        sumatorio += individual[22] * sigmoid(individual[7] * (outputMotor2 + individual[11]))  # Motor 2

        deltaSensor1 = -outputSensor1 + sumatorio + input1
        outputSensor1 = outputSensor1 + ((1 / individual[0]) * deltaSensor1)

        # Sensor 2
        sumatorio = 0.0
        sumatorio += individual[12] * sigmoid(individual[4] * (outputSensor1 + individual[8]))  # Sensor 1
        sumatorio += individual[20] * sigmoid(individual[6] * (outputMotor1 + individual[10]))  # Motor 1
        sumatorio += individual[23] * sigmoid(individual[7] * (outputMotor2 + individual[11]))  # Motor 2

        deltaSensor2 = -outputSensor2 + sumatorio + input2
        outputSensor2 = outputSensor2 + ((1 / individual[1]) * deltaSensor2)

        # print(outputSensor1, " ", outputSensor2)

# ACTUALIZAMOS LOS MOTORES
        # Motor 1
        sumatorio = 0.0
        sumatorio += individual[13] * sigmoid(individual[4] * (outputSensor1 + individual[8]))  # Sensor 1
        sumatorio += individual[16] * sigmoid(individual[5] * (outputSensor2 + individual[9]))  # Sensor 2
        sumatorio += individual[21] * sigmoid(individual[7] * (outputMotor2 + individual[11]))  # Motor 2

        deltaMotor1 = -outputMotor1 + sumatorio
        outputMotor1 = outputMotor1 + ((1 / individual[2]) * deltaMotor1)

        # Motor 2
        sumatorio = 0.0
        sumatorio += individual[14] * sigmoid(individual[4] * (outputSensor1 + individual[8]))  # Sensor 1
        sumatorio += individual[17] * sigmoid(individual[5] * (outputSensor2 + individual[9]))  # Sensor 2
        sumatorio += individual[18] * sigmoid(individual[6] * (outputMotor1 + individual[10]))  # Motor 1

        deltaMotor2 = -outputMotor2 + sumatorio
        outputMotor2 = outputMotor2 + ((1 / individual[3]) * deltaMotor2)

        # print(outputMotor1, " ", outputMotor2)

        # Si alguna de las salidas en NaN, descartamos el agente
        # if (math.isnan(outputMotor1) or math.isnan(outputMotor2)):
        #     return 0

# RECALCULAMOS LA POSICION DEL AGENTE
        # Actualizamos velocidades
        vl = sigmoid(outputMotor1 + individual[10])
        vr = sigmoid(outputMotor2 + individual[11])

        # Esto siempre es un valor entre 0 y 1, lo queremos convertir entre -1 y 1
        vl = (vl * 2.0) - 1.0
        vr = (vr * 2.0) - 1.0

        # print(vl, " ", vr)

        # Multiplicamos por la ganancia del motor
        vl = vl * individual[6]
        vr = vr * individual[7]

        v = (vr + vl) / 2

    # W es - inf o +inf a veces y eso produce NANS
        w = ((vr - vl) / diametroAgente)

        # Recalculamos la posicion del agente
        xAgente += v * math.cos(anguloAgente)
        yAgente += v * math.sin(anguloAgente)

        historialX.append(xAgente)
        historialY.append(yAgente)

        # Recalculamos el angulo del agente
        anguloAgente = normalize(anguloAgente + w)

        ciclos += 1

    plt.plot(xLuz, yLuz, "ro")
    plt.plot(historialX, historialY)
    plt.show()

# W ES NAN A VECES
    # Devolvemos fitness
    # distanciaFinal = math.hypot(xAgente - xLuz, yAgente - yLuz)
    distanciaFinal = distance(xAgente, yAgente, xLuz, yLuz)
    #print(xAgente," - ", yAgente," / ", xLuz, " - ", yLuz, " / ", distanciaFinal)
    # if (distanciaFinal < 10):
    #     print(distanciaFinal)
    if ( distanciaFinal == 0.0):
        return 1
    elif ( distanciaFinal >= 20):
        return 0
    elif ( math.isnan(distanciaFinal)):
        return 0
    else: return 1 / distanciaFinal


# VARIABLES GLOBALES
# En un tablero de 20 x 20
xLuz = 10  # Coordenada X de la luz
yLuz = 10  # Coordenada Y de la luz
DISTANCIA = 10 # Distancia de comienzo del agente respecto a la luz
ITERACIONES = 400 # Numero de iteraciones de la prueba
SEPARACIONSENSOR = 1.0472 # Separacion de lo sensores respecto del eje del angulo del agente, 60º
VISIONSENSOR = 1.39626 # Amplitud angular en la que el sensor recoge lecturas, 80
maxV = 2   # Maxima velocidad lineal permitida
minV = -2  # Minima velocidad lineal permitida
maxW = 2   # Maxima velocidad angular permitida
minW = -2  # Minima velocidad angular permitida

if __name__ == "__main__":

    data = [('xLuz', xLuz), ('yLuz', yLuz)]
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
