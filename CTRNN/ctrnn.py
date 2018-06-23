import random
from random import randint
import matplotlib.pyplot as plt
from pyeasyga import pyeasyga
import math

from deap import base
from deap import creator
from deap import tools

################################ Functions ##################################################################
# Sigmoid function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Angle normalization function
def normalize(angulo):
    return angulo % 2*math.pi

# Function that return the distance between two 2D points
def distance(x1,y1,x2,y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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

#############################################################################################################
################################# Genetic algorithm functions ###############################################
# Custom mutation function for the genetic algorithm
def custom_mutation(individual):
    # Mutate float parts
    for i in range(0, INDIVIDUAL_SIZE - N_NEURONAS):
        if (random.random() > 0.5):
            individual[i] = random.random()

    # Mutate integer parts
    for i in range(INDIVIDUAL_SIZE - N_NEURONAS, INDIVIDUAL_SIZE):
        if (random.random() > 0.9):
            individual[i] = random.randint(0 , 3)

    return individual,

# Function used to create individuals
def initES(icls):
    floats=[random.random() for _ in range(0,INDIVIDUAL_SIZE-N_NEURONAS)]
    ints=[random.randint(0 , 3) for _ in range (0, N_NEURONAS)]
    return icls(floats + ints)

# Evaluation function
def evaluate(individual):
    # Scale normalized [0,1] values to real values
    lastIndex = 0
    vBias = individual[:N_NEURONAS]
    lastIndex = N_NEURONAS
    vTau = individual[lastIndex:lastIndex + N_NEURONAS]
    lastIndex = lastIndex + N_NEURONAS
    gainMotor = individual[lastIndex + 1]
    lastIndex = lastIndex + 1
    gainSensor = individual[lastIndex + 1]
    lastIndex = lastIndex + 1
    tempW = individual[lastIndex: lastIndex + (N_NEURONAS * N_NEURONAS)]
    lastIndex = lastIndex + (N_NEURONAS * N_NEURONAS)
    tempP = individual[lastIndex: lastIndex + (N_NEURONAS * N_NEURONAS)]
    lastIndex = lastIndex + (N_NEURONAS * N_NEURONAS)
    vPType = individual[lastIndex:]

    vBias = [BIAS_MIN + i * (BIAS_MAX - BIAS_MIN) for i in vBias]
    vTau = [TAU_MIN + i * (TAU_MAX - TAU_MIN) for i in vTau]
    gainMotor = GAIN_MIN + gainMotor * (GAIN_MAX - GAIN_MIN)
    gainSensor = GAIN_MIN + gainSensor * (GAIN_MAX - GAIN_MIN)
    tempW = [W_MIN + i * (W_MAX - W_MIN) for i in tempW]
    tempP = [PLASTICIY_RATE_MIN + i * (PLASTICIY_RATE_MAX - PLASTICIY_RATE_MIN) for i in tempP]

    vW = []
    for i in range(0, N_NEURONAS):
        lista = []
        for j in range(0, N_NEURONAS):
            lista.append(tempW[(N_NEURONAS * i) + j])

        vW.append(lista)

    vP = []
    for i in range(0, N_NEURONAS):
        lista = []
        for j in range(0, N_NEURONAS):
            lista.append(tempP[(N_NEURONAS * i) + j])

        vP.append(lista)

    # Preparation of fitness array to calculate total fitness at the end of the run
    fitnessValues = []
    homeostaticControl = [1 for _ in range(0, N_NEURONAS)]
    for i in range(0,N_LUCES):
        fitnessValues.append([0,0,0,0,[]])   # Initial distance, final distance, steps close enought to the light, time that the light has been on, control array of neurons that behave homeostatically
        fitnessValues[i][4] = homeostaticControl[:]

    # Preparation of the rest of the parameters
    inputs = [0.0 for _ in range(0,N_NEURONAS)] # Inputs of the neurons
    outputs = [0.0 for _ in range(0,N_NEURONAS)] # Outputs of the neurons

    xAgente = 0.0
    yAgente = 0.0
    anguloAgente = random.random() * (2 * math.pi)

    # Sensor 1 vision limits
    llimit1 = normalize(SEPARACIONSENSOR + VISIONSENSOR)   # 60º + 20º in radians
    hlimit1 = normalize(SEPARACIONSENSOR - VISIONSENSOR)   # 60º - 80º in radians

    # Sensor 2 vision limits
    llimit2 = normalize(-SEPARACIONSENSOR + VISIONSENSOR)  # -60º + 80º in radians
    hlimit2 = normalize(-SEPARACIONSENSOR - VISIONSENSOR)  # -60º - 20º in radians

    for luces in range(0, N_LUCES):
        distancia = random.uniform(DISTANCIA_LUZ_MIN, DISTANCIA_LUZ_MAX)
        angulo = random.random() * (2 * math.pi)
        xLuz = xAgente + (math.cos(angulo) * distancia) # Light X position
        yLuz = yAgente + (math.sin(angulo) * distancia) # Light Y position
        intensidadLuz = random.uniform(INTENSIDAD_LUZ_MIN, INTENSIDAD_LUZ_MAX)
        time = random.randint(0.75 * T, 1.25 * T)   # Time the light will be on

        # Save initial distance for that light
        fitnessValues[luces][0] = distance(xAgente, yAgente, xLuz, yLuz)

        # Save the time the light will be on
        fitnessValues[luces][3] = time

        # PreIntegration in order to let the individual stabilize, only the first light
        if (luces == 0):
            for ciclos in range(0, CICLOS_PREVIOS):
                inputs[0] = 0.0
                inputs[1] = 0.0

                # Sensor 1 position
                rad1 = normalize(anguloAgente + SEPARACIONSENSOR)
                xSensor1 = xAgente + ((RADIO_AGENTE) * math.cos(rad1))
                ySensor1 = yAgente + ((RADIO_AGENTE) * math.sin(rad1))

                # Sensor 2 position
                rad2 = normalize(anguloAgente - SEPARACIONSENSOR)
                xSensor2 = xAgente + ((RADIO_AGENTE) * math.cos(rad2))
                ySensor2 = yAgente + ((RADIO_AGENTE) * math.sin(rad2))

                # Angle between light and agent
                angAgenteLuz = normalize(math.atan2(yLuz - yAgente, xLuz - xAgente) - anguloAgente)

                # # Check if the sensors will be ON and update inputs
                # if (angAgenteLuz <= llimit1):
                #     ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2
                #     inputs[0] = intensidadLuz / ds1
                #     if (angAgenteLuz <= llimit2):
                #         ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
                #         inputs[1] = intensidadLuz / ds2
                # elif (angAgenteLuz >= hlimit2):
                #     ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
                #     inputs[1] = intensidadLuz / ds2
                #     if (angAgenteLuz >= hlimit1):
                #         ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2
                #         inputs[0] = intensidadLuz / ds1
                # Check if the sensors will be ON and update inputs
                if (angAgenteLuz <= llimit1):
                    # Square of the distance between the light and the sensor
                    ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2
                    # Distance between the light and the center of the agent
                    da = distance(xAgente, yAgente, xLuz, yLuz)
                    a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds1) / (da * da)
                    if (a <= 1.0):
                        inputs[0] = intensidadLuz / ds1
                    if (angAgenteLuz <= llimit2):
                        # Square of the distance between the light and the sensor
                        ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
                        # Distance between the light and the center of the agent
                        da = distance(xAgente, yAgente, xLuz, yLuz)
                        a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds2) / (da * da)
                        if (a <= 1.0):
                            inputs[1] = intensidadLuz / ds2
                elif (angAgenteLuz >= hlimit2):
                    # Square of the distance between the light and the sensor
                    ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
                    # Distance between the light and the center of the agent
                    da = distance(xAgente, yAgente, xLuz, yLuz)
                    a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds2) / (da * da)
                    if (a <= 1.0):
                        inputs[1] = intensidadLuz / ds2
                    if (angAgenteLuz >= hlimit1):
                        # Square of the distance between the light and the sensor
                        ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2
                        # Distance between the light and the center of the agent
                        da = distance(xAgente, yAgente, xLuz, yLuz)
                        a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds1) / (da * da)
                        if (a <= 1.0):
                            inputs[0] = intensidadLuz / ds1

                # Multiply with the gain
                inputs[0] = inputs[0] * gainSensor
                inputs[1] = inputs[1] * gainSensor

                # Make CTRNN RUN
                for i in range(0, N_NEURONAS):
                    change = -outputs[i]

                    for j in range(0, N_NEURONAS):
                        temp = outputs[j] + vBias[j]
                        change += vW[j][i] * sigmoid(temp)

                    change = change + inputs[i]
                    change = change / vTau[i]

                    outputs[i] = outputs[i] + (change * TIME_STEP)

        # Once PreIntegration is finished, we can start our run
        for ciclos in range(0, time):
            inputs[0] = 0.0
            inputs[1] = 0.0

            # Sensor 1 position
            rad1 = normalize(anguloAgente + SEPARACIONSENSOR)
            xSensor1 = xAgente + ((RADIO_AGENTE) * math.cos(rad1))
            ySensor1 = yAgente + ((RADIO_AGENTE) * math.sin(rad1))

            # Sensor 2 position
            rad2 = normalize(anguloAgente - SEPARACIONSENSOR)
            xSensor2 = xAgente + ((RADIO_AGENTE) * math.cos(rad2))
            ySensor2 = yAgente + ((RADIO_AGENTE) * math.sin(rad2))

            # Angle between light and agent
            angAgenteLuz = normalize(math.atan2(yLuz - yAgente, xLuz - xAgente) - anguloAgente)

            # # Check if the sensors will be ON and update inputs
            # if (angAgenteLuz <= llimit1):
            #     ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2
            #     inputs[0] = intensidadLuz / ds1
            #     if (angAgenteLuz <= llimit2):
            #         ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
            #         inputs[1] = intensidadLuz / ds2
            # elif (angAgenteLuz >= hlimit2):
            #     ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
            #     inputs[1] = intensidadLuz / ds2
            #     if (angAgenteLuz >= hlimit1):
            #         ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2
            #         inputs[0] = intensidadLuz / ds1

            # Check if the sensors will be ON and update inputs
            if (angAgenteLuz <= llimit1):
                # Square of the distance between the light and the sensor
                ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2
                # Distance between the light and the center of the agent
                da = distance(xAgente, yAgente, xLuz, yLuz)
                a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds1) / (da * da)
                if (a <= 1.0):
                    inputs[0] = intensidadLuz / ds1
                if (angAgenteLuz <= llimit2):
                    # Square of the distance between the light and the sensor
                    ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
                    # Distance between the light and the center of the agent
                    da = distance(xAgente, yAgente, xLuz, yLuz)
                    a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds2) / (da * da)
                    if (a <= 1.0):
                        inputs[1] = intensidadLuz / ds2
            elif (angAgenteLuz >= hlimit2):
                # Square of the distance between the light and the sensor
                ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
                # Distance between the light and the center of the agent
                da = distance(xAgente, yAgente, xLuz, yLuz)
                a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds2) / (da * da)
                if (a <= 1.0):
                    inputs[1] = intensidadLuz / ds2
                if (angAgenteLuz >= hlimit1):
                    # Square of the distance between the light and the sensor
                    ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2
                    # Distance between the light and the center of the agent
                    da = distance(xAgente, yAgente, xLuz, yLuz)
                    a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds1) / (da * da)
                    if (a <= 1.0):
                        inputs[0] = intensidadLuz / ds1

            # Multiply with the gain
            inputs[0] = inputs[0] * gainSensor
            inputs[0] = inputs[1] * gainSensor

            # Make CTRNN RUN
            for i in range(0, N_NEURONAS):
                change = -outputs[i]

                for j in range(0, N_NEURONAS):
                    temp = outputs[j] + vBias[j]
                    change += vW[j][i] * sigmoid(temp)

                change = change + inputs[i]
                change = change / vTau[i]

                outputs[i] = outputs[i] + (change * TIME_STEP)

            # Allow plasticity changes
            for i in range(0, N_NEURONAS):
                for j in range(0, N_NEURONAS):
                    if vPType != 0:   # If there is no plasticity, nothing is done
                        weight = vW[i][j]
                        jY = outputs[j]
                        jBias = vBias[j]
                        iN = vP[i][j]
                        iRate = sigmoid(outputs[i] + vBias[i])
                        jRate = sigmoid(outputs[j] + vBias[j])

                        jPlastic = plasticity(4.0, 2.0, jY, jBias)

                        delta = 0.0

                        # Check if neuron has gone out of homeostatic bounds
                        if (jPlastic > 0.0):
                            fitnessValues[luces][4][j] = 0

                        damping = W_MAX - math.fabs(weight)

                        if (vPType == 1):
                            delta = damping * iN * jPlastic * iRate * jRate
                        elif (vPType == 2):
                            threshold = (weight + W_MAX) * (W_MAX * 2)
                            delta = damping * iN * jPlastic * (iRate - threshold) * jRate
                        elif (vPType == 3):
                            threshold = (weight + W_MAX) / (W_MAX * 2)
                            delta = damping * iN * jPlastic * iRate * (jRate - threshold)

                        # Weight update
                        weight = weight + delta;
                        if (weight < W_MIN):
                            weight = W_MIN
                        elif (weight > W_MAX):
                            weight = W_MAX

                        # Save update weight
                        vW[i][j] = weight

            # Update speed and position
            vr = sigmoid(outputs[N_NEURONAS-1] + vBias[N_NEURONAS-1])
            vl = sigmoid(outputs[N_NEURONAS-2] + vBias[N_NEURONAS-2])

            # Set speed between -1 and 1 (currently between 0 and 1)
            vr = (vr * 2.0) - 1.0
            vl = (vl * 2.0) - 1.0

            # Multiply with the gain
            vr = vr * gainMotor
            vl = vl * gainMotor

            # Velocities
            v = (vl + vr) / 2.0
            w = (vr - vl) / 2.0 / RADIO_AGENTE

            # Calculate new agent position
            xAgente = xAgente + (v * math.cos(anguloAgente) * TIME_STEP)
            yAgente = yAgente + (v * math.sin(anguloAgente) * TIME_STEP)
            anguloAgente = anguloAgente + (w * TIME_STEP)

            # Check if agent is near the light in this step to add fitness
            if (distance(xAgente, yAgente, xLuz, yLuz) < DISTANCIA_MIN_FITNESS):
                fitnessValues[luces][2] += 1

        # At the end of current light, save final distance
        fitnessValues[luces][1] = distance(xAgente, yAgente, xLuz, yLuz)

    # Rebuild the individual with the new weights
    vBias = [(i - BIAS_MIN)/(BIAS_MAX - BIAS_MIN) for i in vBias]
    vTau = [(i - TAU_MIN)/(TAU_MAX - TAU_MIN) for i in vTau]
    gainMotor = (gainMotor - GAIN_MIN)/(GAIN_MAX - GAIN_MIN)
    gainSensor = (gainSensor - GAIN_MIN)/(GAIN_MAX - GAIN_MIN)
    flatW = [item for sublist in vW for item in sublist]
    flatP = [item for sublist in vP for item in sublist]
    flatW = [(i - W_MIN)/(W_MAX - W_MIN) for i in flatW]
    flatP = [(i - PLASTICIY_RATE_MIN)/(PLASTICIY_RATE_MAX - PLASTICIY_RATE_MIN) for i in flatP]
    indiv = vBias + vTau + [gainMotor] + [gainSensor] + flatW + flatP + vPType
    individual = indiv[:]

    # When all lights have been run, calculate fitness
    finalFitness = []
    for i in range(0, N_LUCES):
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

        finalFitness.append((Fd * 0.34) + (Fp * 0.54) + (Fh * 0.12))
        # finalFitness.append((Fd * 0.2) + (Fp * 0.8))


    return (sum(finalFitness) / N_LUCES),


#############################################################################################################
#################################### Global parameters and constants ########################################
N_NEURONAS_INTERMEDIAS = 0
N_NEURONAS = N_NEURONAS_INTERMEDIAS + 4
INDIVIDUAL_SIZE = 3 * N_NEURONAS + 2 + 2 * (N_NEURONAS * N_NEURONAS)
N_LUCES = 6

CICLOS_PREVIOS = 50

RADIO_AGENTE = 4    # Agent radius
DIAMETRO_AGENTE = RADIO_AGENTE * 2  # Agent diameter
DISTANCIA_LUZ_MIN = RADIO_AGENTE * 10  # Min. distance that the lightsource can appear from the (0,0)
DISTANCIA_LUZ_MAX = RADIO_AGENTE * 25  # Max. distance that the lightsource can appear from the (0,0)
DISTANCIA_MIN_FITNESS = RADIO_AGENTE * 4    # Distance in wich we consider the agent close enough to the lightsource
INTENSIDAD_LUZ_MIN = 500 # Min. value of light intensity
INTENSIDAD_LUZ_MAX = 1500 # Max. value of light intensity
SEPARACIONSENSOR = 1.0472 # Separation between sensor position and agent axis angle, 60º
VISIONSENSOR = 1.39626 # Reading arc of the sensor in wich it reads light inputs, 80

T = 1600 # Variable to calculate the time the lightsource will be on

W_MAX = 10.0
W_MIN = -10.0
TAU_MAX = 4.0
TAU_MIN = 0.4
BIAS_MAX = 3.0
BIAS_MIN = -3.0
GAIN_MAX = 10.0
GAIN_MIN = 0.01
PLASTICIY_RATE_MAX = 0.9
PLASTICIY_RATE_MIN = -0.9

TIME_STEP = 0.2
#############################################################################################################
##################################### MAIN ##################################################################

if __name__ == "__main__":

    creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("individual", initES, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate)
    toolbox.register("crossover", tools.cxUniform)
    # toolbox.register("crossover", tools.cxTwoPoint)
    toolbox.register("mutation", custom_mutation)
    toolbox.register("selection", tools.selTournament, tournsize=10)
    # toolbox.register("selection", tools.selRoulette)

    # Create population
    pop = toolbox.population(n=60)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Extract all fitness of individuals
    fits = [ind.fitness.values[0] for ind in pop]

    # Begin evolution
    generations = 0.0
    while(max(fits) < 0.95):
        generations = generations + 1
        print("Generation number: ", generations)
        print("Best fitness at the moment: ", max(fits))
        print("Best individual at the moment: ", pop[fits.index(max(fits))])
        print("-----------------------------------------")

        # Select next generation of individuals
        offspring = toolbox.selection(pop, len(pop))

        # Clone selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Execute crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if (random.random() < 0.5):
                toolbox.crossover(child1, child2, 0.5)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if (random.random() < 0.5):
                toolbox.mutation(mutant)
                del mutant.fitness.values

        # Re-evaluate individuals with invalid fitnesses (affected by changes)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace old population with offspring
        pop[:] = offspring

        # Extract all fitness of individuals
        fits = [ind.fitness.values[0] for ind in pop]

    # When it finishes, we can print some values and stats
    print("Total number of generations: ", generations)
    bestIndex = fits.index(max(fits))
    bestFitness = fits[bestIndex]
    bestIndividual = pop[bestIndex]
    print(str(bestFitness) + " -> " + str(bestIndividual))
    # Write all population values in a file
    file = open("result.txt","w")
    for i in range(0, len(pop)):
        file.write(str(fits[i]) + " -> " + str(pop[i]) + "\n")
