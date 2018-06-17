import matplotlib.pyplot as plt
from pyeasyga import pyeasyga
from random import randint

import random
import math

# Sigmoid function
def sigmoid_math(x):
  return 1 / (1 + math.exp(-x))

# Plasticity check, as used in "On Adapation via Ultrastability", Candiate Number: 52804
def plasticity(max, min, y, b):
    activity = y + b
    p = 0.0
    if (activity < -max):
        p = -1
    elif (activity > max):
        p = 1
    elif (activity < -min):
        p = (0.5 * activity) + 1
    elif (activity > min):
        p = (0.5 * activity) - 1

    return p

# Angle normalization function
def normalize(angulo):
    return angulo % 2*math.pi

# Function that return the distance between two 2D points
def distance(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # return math.hypot(x2 - x1, y2 - y1)

# PreRun: 50 cicles of pre execution to let the agent stabilize
def preIntegrate(individual):
    inputs = []
    outputs = []

    for i in range(0, N_NEURONAS):
        outputs.append(0.0)
        inputs.append(0.0)

    for i in range(0, 50):
        for i in range(0, N_NEURONAS):
            change = -1.0 * outputs[i]

            for j in range(0, N_NEURONAS):
                temp = outputs[j] + individual[2][j]
                change += individual[3][j][i] * sigmoid_math(temp)

            # Read neuron input
            change +=inputs[i]

            # Tau factor application
            change = change / individual[0][i]

            # Save changes
            outputs[i] = outputs[i] + (change * TIME_STEP)

    return outputs

# Crossover function
def crossover(parent_1, parent_2):
    crossover_index_N = random.randrange(0, N_NEURONAS)
    crossover_index_N2 = random.randrange(0, N_NEURONAS)

    child_1 = parent_1[:]
    child_2 = parent_2[:]

    # Cross taus, gains and biases
    for i in [0,1,2]:
        child_1[i] = parent_1[i][:crossover_index_N] + parent_2[i][crossover_index_N:]
        child_2[i] = parent_2[i][:crossover_index_N] + parent_1[i][crossover_index_N:]

    # We ensure the gain simetry in sensor and motor neurons
    child_1[1][1] = child_1[1][0]
    child_2[1][1] = child_2[1][0]
    child_1[1][3] = child_1[1][2]
    child_2[1][3] = child_2[1][2]

    # Cross weights
    for i in range(0, N_NEURONAS):
        child_1[3][i] = parent_1[3][i][:crossover_index_N2] + parent_2[3][i][crossover_index_N2:]
        child_2[3][i] = parent_2[3][i][:crossover_index_N2] + parent_1[3][i][crossover_index_N2:]

    # Cross plasticity factors
    for i in range(0, N_NEURONAS):
        child_1[4][i] = parent_1[4][i][:crossover_index_N2] + parent_2[4][i][crossover_index_N2:]
        child_2[4][i] = parent_2[4][i][:crossover_index_N2] + parent_1[4][i][crossover_index_N2:]

    # Cross plasticity types
    child_1[5] = parent_1[5][:crossover_index_N] + parent_2[5][crossover_index_N:]
    child_2[5] = parent_2[5][:crossover_index_N] + parent_1[5][crossover_index_N:]

    return child_1, child_2

# Function that selects a random individual from the genetic population
def selection(population):
    return random.choice(population)

# Mutation functions. Takes a random index and mutates that value
def mutate(individual):
    index_affected = random.randrange(0,6)

    if (index_affected == 0):
        index = random.randrange(0, N_NEURONAS)
        individual[0][index] = round(random.uniform(TAU_MIN,TAU_MAX),4)
    elif (index_affected == 1):
        index = random.randrange(0, 4)
        randomValue = round(random.uniform(GAIN_MIN, GAIN_MAX),4)
        if (index == 0) or (index == 1):
            individual[1][0] = randomValue
            individual[1][1] = randomValue
        else:
            individual[1][2] = randomValue
            individual[1][3] = randomValue
    elif (index_affected == 2):
        index = random.randrange(0, N_NEURONAS)
        individual[2][index] = round(random.uniform(BIAS_MIN,BIAS_MAX),4)
    elif (index_affected == 3):
        index_inside = random.randrange(0, N_NEURONAS)
        index_inside_2 = random.randrange(0, N_NEURONAS)
        randomValue = round(random.uniform(W_MIN,W_MAX),4)
        individual[3][index_inside][index_inside_2] = randomValue
    elif (index_affected == 4):
        index_inside = random.randrange(0, N_NEURONAS)
        index_inside_2 = random.randrange(0, N_NEURONAS)
        randomValue = round(random.uniform(PLASTICIY_RATE_MIN,PLASTICIY_RATE_MAX),4)
        individual[4][index_inside][index_inside_2] = randomValue
    elif (index_affected == 5):
        index = random.randrange(0, N_NEURONAS)
        individual[5][index] = round(randint(0,3),4)

# Function that creates an individual for the population
def create_individual(data):
    individual = []

    # Append tau
    vectorTau = []
    vectorTau.append(round(random.uniform(TAU_MIN,TAU_MAX),4)) #  tau sensor 1
    vectorTau.append(round(random.uniform(TAU_MIN,TAU_MAX),4)) #  tau sensor 2

    for i in range (0, N_INTERMEDIAS):
        vectorTau.append(round(random.uniform(TAU_MIN,TAU_MAX),4)) #  intermediate neurons tau

    vectorTau.append(round(random.uniform(TAU_MIN,TAU_MAX),4)) #  tau motor 1
    vectorTau.append(round(random.uniform(TAU_MIN,TAU_MAX),4)) # tau motor 2

    individual.append(vectorTau)

    # Append gains
    vectorGain = []
    gainSensor = round(random.uniform(GAIN_MIN, GAIN_MAX),4)
    vectorGain.append(gainSensor)  # gain sensor 1
    vectorGain.append(gainSensor)  # gain sensor 2

    gainMotor = round(random.uniform(GAIN_MIN,GAIN_MAX),2)
    vectorGain.append(gainMotor)  # gain motor 1
    vectorGain.append(gainMotor)  # gain motor 2

    individual.append(vectorGain)

    # Append bias
    vectorBias = []
    vectorBias.append(round(random.uniform(BIAS_MIN,BIAS_MAX),4))  # bias sensor 1
    vectorBias.append(round(random.uniform(BIAS_MIN,BIAS_MAX),4))  # bias sensor 2

    for i in range (0, N_INTERMEDIAS):
        vectorBias.append(round(random.uniform(BIAS_MIN,BIAS_MAX),4)) # intermediate neurons bias

    vectorBias.append(round(random.uniform(BIAS_MIN,BIAS_MAX),4))  # bias motor 1
    vectorBias.append(round(random.uniform(BIAS_MIN,BIAS_MAX),4))  # bias motor 2

    individual.append(vectorBias)

    # Append weights
    vectorPesos = []
    for i in range(0, N_NEURONAS):
        vectorPesosNeurona = []
        for j in range(0, N_NEURONAS):
            vectorPesosNeurona.append(round(random.uniform(W_MIN,W_MAX),4))
        vectorPesos.append(vectorPesosNeurona)

    individual.append(vectorPesos)

    # Append plasticity factor
    vectorPlasticidad = []
    for i in range(0, N_NEURONAS):
        vectorPlasticidadNeurona = []
        for j in range(0, N_NEURONAS):
            vectorPlasticidadNeurona.append(round(random.uniform(PLASTICIY_RATE_MIN,PLASTICIY_RATE_MAX),4))
        vectorPlasticidad.append(vectorPlasticidadNeurona)

    individual.append(vectorPlasticidad)

    # Append plasticity type
    vectorTipoP = []
    vectorTipoP.append(round(randint(0,3),4))
    vectorTipoP.append(round(randint(0,3),4))

    for i in range (0, N_INTERMEDIAS):
        vectorTipoP.append(round(randint(0,3),4))

    vectorTipoP.append(round(randint(0,3),4))
    vectorTipoP.append(round(randint(0,3),4))

    individual.append(vectorTipoP)

    # INDEX OF AN INDIVIDUAL:
    #   0 = taus
    #   1 = gains
    #   2 = bias
    #   3 = weights (2D vector, i neuron with j neuron, etc)
    #   4 = plasticity factor (2D vector, i neuron with j neuron, etc)
    #   5 = plasticity type (0 = no plasticity, 1 = bounded hebbian lerning, 2= Dampen potentiation or depression of the presynaptic neuron when the synaptic efficicy is too high or
    #                       too low, 3= Dampen potentiation or depression of the postsynaptic neuron when the synaptic efficicy is too high or too low)

    return individual

def fitness(individual,data):
    # Agent starts at (0,0) and with a random orientation
    xAgente = 0.0
    yAgente = 0.0
    anguloAgente = random.random() * (2 * math.pi)

    # Inputs of the neurons
    inputs = []

    # Outputs of the neurons
    outputs = preIntegrate(individual)

    for i in range(0, N_NEURONAS):
        # outputs.append(0.0)
        inputs.append(0.0)

    # Preparation of fitness array to calculate total fitness at the end of the run
    fitnessValues = []
    homeostaticControl = []
    for i in range(0, N_NEURONAS):
        homeostaticControl.append(1)
    for i in range(0,6):
        fitnessValues.append([0,0,0,0,[],0,0])   # Initial distance, final distance, steps close enought to the light, time that the light has been on, control array of neurons that behave homeostatically, final agentX, final agentY
        fitnessValues[i][4] = homeostaticControl[:]

    # Creation of the lights that will participate in the experiment
    lucesX = []
    lucesY = []
    for i in range(0,6):

        angulo = random.random() * (2 * math.pi)
        distancia = round(random.uniform(DISTANCIA_LUZ_MIN, DISTANCIA_LUZ_MAX),2)
        lucesX.append(xAgente + (math.cos(angulo) * distancia)) # Light X position
        lucesY.append(yAgente + (math.sin(angulo) * distancia)) # Light Y position

## BEGINING OF THE EXPERIMENT

    historialX = []
    historialY = []
    endAgenteX = []
    endAgenteY = []
    for luces in range(0,6):
        xLuz = lucesX[luces]
        yLuz = lucesY[luces]
        intensidadLuz = random.randint(INTENSIDAD_LUZ_MIN, INTENSIDAD_LUZ_MAX)
        tiempoEncendida = random.randint(0.75 * T, 1.25 * T)   # Time the light will be on

        # Save initial distance between agent and that ligthsource
        fitnessValues[luces][0] = distance(xAgente, yAgente, xLuz, yLuz)

        # Save the time light will be online
        fitnessValues[luces][3] = tiempoEncendida

        for ciclos in range(0, tiempoEncendida):
            inputs[0] = 0.0
            inputs[1] = 0.0

            # Angle between light and agent
            angAgenteLuz = normalize(math.atan2(yLuz - yAgente, xLuz - xAgente) - anguloAgente)

            # Sensor 1 vision limits
            llimit1 = normalize(SEPARACIONSENSOR + VISIONSENSOR)   # 60º + 20º in radians
            hlimit1 = normalize(SEPARACIONSENSOR - VISIONSENSOR)   # 60º - 80º in radians

            # Sensor 2 vision limits
            llimit2 = normalize(-SEPARACIONSENSOR + VISIONSENSOR)  # -60º + 80º in radians
            hlimit2 = normalize(-SEPARACIONSENSOR - VISIONSENSOR)  # -60º - 20º in radians

            # Sensor 1 position
            rad1 = normalize(anguloAgente + SEPARACIONSENSOR)
            xSensor1 = xAgente + ((RADIO_AGENTE) * math.cos(rad1))
            ySensor1 = yAgente + ((RADIO_AGENTE) * math.sin(rad1))

            # Sensor 2 position
            rad2 = normalize(anguloAgente - SEPARACIONSENSOR)
            xSensor2 = xAgente + ((RADIO_AGENTE) * math.cos(rad2))
            ySensor2 = yAgente + ((RADIO_AGENTE) * math.sin(rad2))

        # SENSOR UPDATE
            if (angAgenteLuz <= llimit1):
                ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2
                inputs[0] = intensidadLuz / ds1
                if (angAgenteLuz <= llimit2):
                    ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
                    inputs[1] = intensidadLuz / ds2
            elif (angAgenteLuz >= hlimit2):
                ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
                inputs[1] = intensidadLuz / ds2
                if (angAgenteLuz >= hlimit1):
                    ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2
                    inputs[0] = intensidadLuz / ds1

            # First we check if the sensor are active
            # s1Active = False
            # s2Active = False

            # if (angAgenteLuz <= llimit1):
            #     s1Active = True
            #     if (angAgenteLuz <= llimit2):
            #         s2Active = True
            # elif (angAgenteLuz >= hlimit2):
            #     s2Active = True
            #     if (angAgenteLuz >= hlimit1):
            #         s1Active = True
            #
            # # If they are active we calculate they input
            # if (s1Active == True):
            #
            #
            #     # Square of the distance between the light and the sensor
            #     ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2
            #
            #     # Distance between the light and the center of the agent
            #     da = distance(xAgente, yAgente, xLuz, yLuz)
            #
            #     a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds1) / (da * da)
            #
            #     if (a <= 1.0):
            #         inputs[0] = intensidadLuz / ds1
            #
            #
            # if (s2Active == True):
            #
            #
            #     # Square of the distance between the light and the sensor
            #     ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
            #
            #     # Distance between the light and the center of the agent
            #     da = distance(xAgente, yAgente, xLuz, yLuz)
            #
            #     a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds2) / (da * da)
            #
            #     if (a <= 1.0):
            #         inputs[1] = intensidadLuz / ds2

            # Multiply the input for the gain of the sensors
            inputs[0] = inputs[0] * math.exp(individual[1][0])
            inputs[1] = inputs[1] * math.exp(individual[1][1])

            # Make one run of the CTRNN function as used in "On Adapation via Ultrastability", Candiate Number: 52804
            for i in range(0, N_NEURONAS):
                change = -1.0 * outputs[i]

                for j in range(0, N_NEURONAS):
                    temp = outputs[j] + individual[2][j]
                    change += individual[3][j][i] * sigmoid_math(temp)

                # Read neuron input
                change +=inputs[i]

                # Tau factor application
                change = change / individual[0][i]

                # Save changes
                outputs[i] = outputs[i] + (change * TIME_STEP)

            # Update (or not) of the plasticity of the neurons depending on their plasticity type
            for i in range(0, N_NEURONAS):
                for j in range(0, N_NEURONAS):
                    if individual[5][j] != 0:   # If there is no plasticity, nothing is done
                        weight = individual[3][i][j]
                        jY = outputs[j]
                        jBias = individual[2][j]
                        iN = individual[4][i][j]
                        iRate = sigmoid_math(outputs[i] + individual[2][i])
                        jRate = sigmoid_math(outputs[j] + individual[2][j])

                        jPlastic = plasticity(4.0, 2.0, jY, jBias)

                        delta = 0.0

                        # Check if neuron has gone out of homeostatic bounds
                        if (jPlastic > 0.0):
                            fitnessValues[luces][4][j] = 0

                        damping = W_MAX - math.fabs(weight)

                        if (individual[5][j] == 1):
                            delta = damping * iN * jPlastic * iRate * jRate
                        elif (individual[5][j] == 2):
                            threshold = (weight + W_MAX) * (W_MAX * 2)
                            delta = damping * iN * jPlastic * (iRate - threshold) * jRate
                        elif (individual[5][j] == 3):
                            threshold = (weight + W_MAX) / (W_MAX * 2)
                            delta = damping * iN * jPlastic * iRate * (jRate - threshold)

                        # Weight update
                        weight = weight + delta;
                        if (weight < W_MIN):
                            weight = W_MIN
                        elif (weight > W_MAX):
                            weight = W_MAX

                        # Save update weight
                        individual[3][i][j] = weight

            # UPDATE OF MOTOR VALUES AND AGENT POSITION
            vl = sigmoid_math(outputs[N_NEURONAS - 1] + individual[2][N_NEURONAS - 1])
            vr = sigmoid_math(outputs[N_NEURONAS - 2] + individual[2][N_NEURONAS - 2])

            # Value mapping between -1 and 1
            vl = (vl * 2.0) - 1.0
            vr = (vr * 2.0) - 1.0

            # Multiply the motor power for the gain of the motors
            vl = vl * individual[1][3]
            vr = vr * individual[1][2]

            v = (vl + vr) / 2.0
            w = (vr - vl) / (2.0 * RADIO_AGENTE)

            # Save last position
            lastX = xAgente
            lastY = yAgente

            # Calculate change in this time step
            yChange = TIME_STEP * v * math.sin(anguloAgente)
            xChange = TIME_STEP * v * math.cos(anguloAgente)
            aChange = TIME_STEP * w

            # Recalculate agent position
            xAgente = xAgente + xChange
            yAgente = yAgente + yChange
            anguloAgente = normalize(anguloAgente + aChange)

            historialX.append(xAgente)
            historialY.append(yAgente)


            # Check if the agent is close enough to the active lightsource
            if (distance(xAgente, yAgente, xLuz, yLuz) < DISTANCIA_MIN_FITNESS):
                # # Forzamos que se muevan
                # if (distance(xAgente, yAgente, lastX, lastY) >= 2):
                #     fitnessValues[luces][2] += 1
                fitnessValues[luces][2] += 1

        # Save final distance between the agent and the lightsource
        fitnessValues[luces][1] = distance(xAgente, yAgente, xLuz, yLuz)

        # Agent coords at the end of the run for that lightsource
        endAgenteX.append(xAgente)
        endAgenteY.append(yAgente)

# EVALUATE FITNESS OF THE AGENT
    fitnessEvaluator = []
    overall = 0.0
    for i in range(0, 6):
        if (fitnessValues[i][1] > fitnessValues[i][0]):
            Fd = 0.0
        else:
            Fd = 1 - (fitnessValues[i][1] / fitnessValues[i][0])

        Fp = fitnessValues[i][2] / fitnessValues[i][3]

        homeostaticas = 0
        for j in range(0, N_NEURONAS):
            if (fitnessValues[i][4][j] == 1):
                homeostaticas += 1

        Fh = homeostaticas / N_NEURONAS

        fitnessEvaluator.append((Fd * 0.2) + (Fp * 0.8) )

    # Calculate mean of fitness
    mean = sum(fitnessEvaluator) / 6

    # Calculate standard deviation
    differences = [x - mean for x in fitnessEvaluator]
    sq_differences = [d**2 for d in differences]
    ssd = sum(sq_differences)
    variance = ssd / 6
    sd = math.sqrt(variance)

    # Adjust mean by substracting deviation * 0.2
    overall = mean - (sd * 0.2)
    # print(mean)
    # plt.scatter(lucesX, lucesY, s=60, c='red', marker='^')
    # plt.scatter(endAgenteX, endAgenteY, s=30, c='blue', marker='o')
    # for i in range(0, len(lucesX)):
    #     plt.annotate(i, (lucesX[i],lucesY[i]))
    #     plt.annotate(i, (endAgenteX[i],endAgenteY[i]))
    # plt.plot(historialX[0], historialY[0], "bo")
    # plt.plot(historialX, historialY)
    # plt.show()
    return 1.0 - overall

    # Fd = 0.0
    # for k in range(0, 6):
    #     # print(k,": ",fitnessValues[k][0], " ", fitnessValues[k][1])
    #
    #     if (fitnessValues[k][1] > fitnessValues[k][0]):
    #         d = 0.0
    #     else:
    #         d = 1 - (fitnessValues[k][1] / fitnessValues[k][0])
    #     Fd += d
    #
    # Fd = Fd / 6     # Average distance between the agent final position and the lightsources
    #
    # tiempoTotal = 0
    # Fp = 0
    #
    # for k in range(0, 6):
    #     tiempoTotal += fitnessValues[k][3]
    #     Fp += fitnessValues[k][2]
    #
    # previo = Fp
    # # Fp = Fp / tiempoTotal / TIME_STEP
    # Fp = Fp / tiempoTotal
    #
    # homeostaticas = 0
    # h = []
    # for i in range(0, N_NEURONAS):
    #     h.append(1.0)
    #
    # for i in range(0, N_NEURONAS):
    #     for j in range(0,6):
    #         h[i] = h[i] * fitnessValues[j][4][i]
    #
    # for i in range(0, N_NEURONAS):
    #     if (h[i] == 1):
    #         homeostaticas += 1
    #
    # # Fh = homeostaticas / (N_NEURONAS * (tiempoTotal / TIME_STEP))
    # Fh = homeostaticas / (N_NEURONAS)
    # # print("---------------------------------- Fd: ", Fd)
    # # print("Fp: ", previo," -- Time: ", tiempoTotal)
    # # print("---------------------------------- Fp: ", Fp)
    # # for i in range(0,6):
    # #     print(i,": Homeostatic: ", fitnessValues[i][4])
    # #
    # # print("Homeostatic number: ", homeostaticas)
    # # print("---------------------------------- Fh: ", Fh)
    # #
    # # print((Fd * 0.4) + (Fp * 0.6) + (Fh * 0.16),": ", Fd," ",Fp, " ", Fh)
    # # print("--------------------------------------------------------------------------------------")
    # # plt.scatter(lucesX, lucesY, s=60, c='red', marker='^')
    # # plt.scatter(endAgenteX, endAgenteY, s=30, c='blue', marker='o')
    # # for i in range(0, len(lucesX)):
    # #     plt.annotate(i, (lucesX[i],lucesY[i]))
    # #     plt.annotate(i, (endAgenteX[i],endAgenteY[i]))
    # # plt.plot(historialX[0], historialY[0], "bo")
    # # plt.plot(historialX, historialY)
    # # plt.show()
    #
    # # Fitness function is: 0.44 * Average distance between the agent final position and the lightsources + 0.44 * number of timesteps that the agent has been close enough to a lightsource / total of timesteps + 0.12 * number of neurons that havent lost homeostatic behaviour
    # return ((Fd * 0.4) + (Fp * 0.4) + (Fh * 0.2))


# GLOBAL VARIABLES
RADIO_AGENTE = 4    # Agent radius
DIAMETRO_AGENTE = RADIO_AGENTE * 2  # Agent diameter
DISTANCIA_LUZ_MIN = RADIO_AGENTE * 10  # Min. distance that the lightsource can appear from the (0,0)
DISTANCIA_LUZ_MAX = RADIO_AGENTE * 25  # Max. distance that the lightsource can appear from the (0,0)
DISTANCIA_MIN_FITNESS = RADIO_AGENTE * 4    # Distance in wich we consider the agent close enough to the lightsource
INTENSIDAD_LUZ_MIN = 500 # Min. value of light intensity
INTENSIDAD_LUZ_MAX = 1500 # Max. value of light intensity
SEPARACIONSENSOR = 1.0472 # Separation between sensor position and agent axis angle, 60º
VISIONSENSOR = 1.39626 # Reading arc of the sensor in wich it reads light inputs, 80
TIME_STEP = 0.2 # Integration time-step
N_INTERMEDIAS = 0
N_NEURONAS = N_INTERMEDIAS + 4
T = 800 # Variable to calculate the time the lightsource will be on

W_MAX = 10.0
W_MIN = -10.0
TAU_MAX = 4.0
TAU_MIN = 0.4
BIAS_MAX = 3.0
BIAS_MIN = -3.0
GAIN_MIN = 0.01
GAIN_MAX = 10.0

PLASTICIY_RATE_MAX = 0.9
PLASTICIY_RATE_MIN = -0.9

if __name__ == "__main__":

    data = [('xLuz', 0), ('yLuz', 0)]
    ga = pyeasyga.GeneticAlgorithm(data,
                                    population_size=60,
                                    generations=1000,
                                    crossover_probability=0.5,
                                    mutation_probability=0.3,
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

    file = open("result.txt","w")
    file.write(str(ga.best_individual()))
