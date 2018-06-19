import random
from random import randint
import matplotlib.pyplot as plt
from pyeasyga import pyeasyga
import math
import copy

from deap import base
from deap import creator
from deap import tools
################################ Agent class ################################################################
class Agent:
    def __init__(self, posX, posY, angulo, bias, tau, motorGain, sensorGain, vW, vP, vPType):
        self.posX = copy.deepcopy(posX)
        self.posY = copy.deepcopy(posY)
        self.angulo = copy.deepcopy(angulo)
        self.vBias = copy.deepcopy(bias)
        self.vTau = copy.deepcopy(tau)
        self.gainMotor = copy.deepcopy(motorGain)
        self.gainSensor = copy.deepcopy(sensorGain)
        self.vW = copy.deepcopy(vW)
        self.vP = copy.deepcopy(vP)
        self.vPType = copy.deepcopy(vPType)

        self.inputs = [0.0 for _ in range(0,N_NEURONAS)] # Inputs of the neurons
        self.outputs = [0.0 for _ in range(0,N_NEURONAS)] # Outputs of the neurons

        temp = [1 for _ in range(0, N_NEURONAS)]
        self.homeostatic = [ temp[:] for _ in range(0, N_LUCES)]
        self.stepsClose = [0.0 for _ in range(0, N_LUCES)]

        self.v = 0.0
        self.w = 0.0

#############################################################################################################
################################ Functions ##################################################################
# Calculates X position of the agents centroid
def centroidX(agents):
    centX = 0.0
    for agent in agents:
        centX += agent.posX

    return centX / N_AGENTES

# Calculates Y position of the agents centroid
def centroidY(agents):
    centY = 0.0
    for agent in agents:
        centY += agent.posY

    return centY / N_AGENTES

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
    for individuo in range(0, N_AGENTES):
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
    group = []
    for individuo in range(0, N_AGENTES):
        floats=[random.random() for _ in range(0,INDIVIDUAL_SIZE-N_NEURONAS)]
        ints=[random.randint(0 , 3) for _ in range (0, N_NEURONAS)]
        group.extend(floats + ints)
    return icls(group)

# Evaluation function
def evaluate(individual):
    # Scale normalized [0,1] values to real values
    vAgents = []
    for _ in range(0, N_AGENTES):
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
        randomAngle = random.random() * (2 * math.pi)
        randomDistance = random.uniform(DISTANCIA_INICIAL_AGENTE_MIN, DISTANCIA_INICIAL_AGENTE_MAX)
        vAgents.append(Agent(0.0 + (math.cos(randomAngle) * randomDistance), 0.0 + (math.sin(randomAngle) * randomDistance), random.random() * (2 * math.pi), vBias, vTau, gainMotor, gainSensor, vW, vP, vPType))

    # Preparation of fitness array to calculate centroid fitness (each agent takes care of his own fitness)
    fitnessValues = []
    for i in range(0,N_LUCES):
        fitnessValues.append([0.0,0.0, 0.0])   # Initial centroid distance, final centroid distance, time the light has been on

    # Sensor 1 vision limits
    llimit1 = normalize(SEPARACIONSENSOR + VISIONSENSOR)   # 60º + 20º in radians
    hlimit1 = normalize(SEPARACIONSENSOR - VISIONSENSOR)   # 60º - 80º in radians

    # Sensor 2 vision limits
    llimit2 = normalize(-SEPARACIONSENSOR + VISIONSENSOR)  # -60º + 80º in radians
    hlimit2 = normalize(-SEPARACIONSENSOR - VISIONSENSOR)  # -60º - 20º in radians

    for luces in range(0, N_LUCES):
        distancia = random.uniform(DISTANCIA_LUZ_MIN, DISTANCIA_LUZ_MAX)
        angulo = random.random() * (2 * math.pi)
        xLuz = centroidX(vAgents) + (math.cos(angulo) * distancia) # Light X position
        yLuz = centroidY(vAgents) + (math.sin(angulo) * distancia) # Light Y position
        intensidadLuz = random.uniform(INTENSIDAD_LUZ_MIN, INTENSIDAD_LUZ_MAX)
        time = random.randint(0.75 * T, 1.25 * T)   # Time the light will be on

        # Save initial distance for that light
        fitnessValues[luces][0] = distance(centroidX(vAgents), centroidY(vAgents), xLuz, yLuz)

        # Save the time the light will be on
        fitnessValues[luces][2] = time

        # PreIntegration in order to let the individual stabilize
        for ciclos in range(0, CICLOS_PREVIOS):
            for agent in vAgents:
                agent.inputs[0] = 0.0
                agent.inputs[1] = 0.0

                # Sensor 1 position
                rad1 = normalize(agent.angulo + SEPARACIONSENSOR)
                xSensor1 = agent.posX + ((RADIO_AGENTE) * math.cos(rad1))
                ySensor1 = agent.posY + ((RADIO_AGENTE) * math.sin(rad1))

                # Sensor 2 position
                rad2 = normalize(agent.angulo - SEPARACIONSENSOR)
                xSensor2 = agent.posX + ((RADIO_AGENTE) * math.cos(rad2))
                ySensor2 = agent.posY + ((RADIO_AGENTE) * math.sin(rad2))

                # Angle between light and agent
                angAgenteLuz = normalize(math.atan2(yLuz - agent.posY, xLuz - agent.posX) - agent.angulo)

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
                    da = distance(agent.posX, agent.posY, xLuz, yLuz)
                    a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds1) / (da * da)
                    if (a <= 1.0):
                        agent.inputs[0] = intensidadLuz / ds1
                    if (angAgenteLuz <= llimit2):
                        # Square of the distance between the light and the sensor
                        ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
                        # Distance between the light and the center of the agent
                        da = distance(agent.posX, agent.posY, xLuz, yLuz)
                        a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds2) / (da * da)
                        if (a <= 1.0):
                            agent.inputs[1] = intensidadLuz / ds2
                elif (angAgenteLuz >= hlimit2):
                    # Square of the distance between the light and the sensor
                    ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
                    # Distance between the light and the center of the agent
                    da = distance(agent.posX, agent.posY, xLuz, yLuz)
                    a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds2) / (da * da)
                    if (a <= 1.0):
                        agent.inputs[1] = intensidadLuz / ds2
                    if (angAgenteLuz >= hlimit1):
                        # Square of the distance between the light and the sensor
                        ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2
                        # Distance between the light and the center of the agent
                        da = distance(agent.posX, agent.posY, xLuz, yLuz)
                        a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds1) / (da * da)
                        if (a <= 1.0):
                            agent.inputs[0] = intensidadLuz / ds1

                # Multiply with the gain
                agent.inputs[0] = agent.inputs[0] * agent.gainSensor
                agent.inputs[0] = agent.inputs[1] * agent.gainSensor

                # Make CTRNN RUN
                for i in range(0, N_NEURONAS):
                    change = -agent.outputs[i]

                    for j in range(0, N_NEURONAS):
                        temp = agent.outputs[j] + agent.vBias[j]
                        change += agent.vW[j][i] * sigmoid(temp)

                    change = change + agent.inputs[i]
                    change = change / agent.vTau[i]

                    agent.outputs[i] = agent.outputs[i] + (change * TIME_STEP)


        # Once PreIntegration is finished, we can start our run
        for ciclos in range(0, time):
            for agent in vAgents:
                agent.inputs[0] = 0.0
                agent.inputs[1] = 0.0

                # Sensor 1 position
                rad1 = normalize(agent.angulo + SEPARACIONSENSOR)
                xSensor1 = agent.posX + ((RADIO_AGENTE) * math.cos(rad1))
                ySensor1 = agent.posY + ((RADIO_AGENTE) * math.sin(rad1))

                # Sensor 2 position
                rad2 = normalize(agent.angulo - SEPARACIONSENSOR)
                xSensor2 = agent.posX + ((RADIO_AGENTE) * math.cos(rad2))
                ySensor2 = agent.posY + ((RADIO_AGENTE) * math.sin(rad2))

                # Angle between light and agent
                angAgenteLuz = normalize(math.atan2(yLuz - agent.posY, xLuz - agent.posX) - agent.angulo)

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
                    da = distance(agent.posX, agent.posY, xLuz, yLuz)
                    a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds1) / (da * da)
                    if (a <= 1.0):
                        agent.inputs[0] = intensidadLuz / ds1
                    if (angAgenteLuz <= llimit2):
                        # Square of the distance between the light and the sensor
                        ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
                        # Distance between the light and the center of the agent
                        da = distance(agent.posX, agent.posY, xLuz, yLuz)
                        a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds2) / (da * da)
                        if (a <= 1.0):
                            agent.inputs[1] = intensidadLuz / ds2
                elif (angAgenteLuz >= hlimit2):
                    # Square of the distance between the light and the sensor
                    ds2 = distance(xSensor2, ySensor2, xLuz, yLuz)**2
                    # Distance between the light and the center of the agent
                    da = distance(agent.posX, agent.posY, xLuz, yLuz)
                    a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds2) / (da * da)
                    if (a <= 1.0):
                        agent.inputs[1] = intensidadLuz / ds2
                    if (angAgenteLuz >= hlimit1):
                        # Square of the distance between the light and the sensor
                        ds1 = distance(xSensor1, ySensor1, xLuz, yLuz)**2
                        # Distance between the light and the center of the agent
                        da = distance(agent.posX, agent.posY, xLuz, yLuz)
                        a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds1) / (da * da)
                        if (a <= 1.0):
                            agent.inputs[0] = intensidadLuz / ds1

                # Multiply with the gain
                agent.inputs[0] = agent.inputs[0] * agent.gainSensor
                agent.inputs[0] = agent.inputs[1] * agent.gainSensor

                # Make CTRNN RUN
                for i in range(0, N_NEURONAS):
                    change = -agent.outputs[i]

                    for j in range(0, N_NEURONAS):
                        temp = agent.outputs[j] + agent.vBias[j]
                        change += agent.vW[j][i] * sigmoid(temp)

                    change = change + agent.inputs[i]
                    change = change / agent.vTau[i]

                    agent.outputs[i] = agent.outputs[i] + (change * TIME_STEP)

                # Allow plasticity changes
                for i in range(0, N_NEURONAS):
                    for j in range(0, N_NEURONAS):
                        if agent.vPType != 0:   # If there is no plasticity, nothing is done
                            weight = agent.vW[i][j]
                            jY = agent.outputs[j]
                            jBias = agent.vBias[j]
                            iN = agent.vP[i][j]
                            iRate = sigmoid(agent.outputs[i] + agent.vBias[i])
                            jRate = sigmoid(agent.outputs[j] + agent.vBias[j])

                            jPlastic = plasticity(4.0, 2.0, jY, jBias)

                            delta = 0.0

                            # Check if neuron has gone out of homeostatic bounds
                            if (jPlastic > 0.0):
                                agent.homeostatic[luces][j] = 0

                            damping = W_MAX - math.fabs(weight)

                            if (agent.vPType == 1):
                                delta = damping * iN * jPlastic * iRate * jRate
                            elif (agent.vPType == 2):
                                threshold = (weight + W_MAX) * (W_MAX * 2)
                                delta = damping * iN * jPlastic * (iRate - threshold) * jRate
                            elif (agent.vPType == 3):
                                threshold = (weight + W_MAX) / (W_MAX * 2)
                                delta = damping * iN * jPlastic * iRate * (jRate - threshold)

                            # Weight update
                            weight = weight + delta;
                            if (weight < W_MIN):
                                weight = W_MIN
                            elif (weight > W_MAX):
                                weight = W_MAX

                            # Save update weight
                            agent.vW[i][j] = weight

                # Update speed and position
                vr = sigmoid(agent.outputs[N_NEURONAS-1] + agent.vBias[N_NEURONAS-1])
                vl = sigmoid(agent.outputs[N_NEURONAS-2] + agent.vBias[N_NEURONAS-2])

                # Set speed between -1 and 1 (currently between 0 and 1)
                vr = (vr * 2.0) - 1.0
                vl = (vl * 2.0) - 1.0

                # Multiply with the gain
                vr = vr * gainMotor
                vl = vl * gainMotor

                # Velocities
                agent.v = (vl + vr) / 2.0
                agent.w = (vr - vl) / 2.0 / RADIO_AGENTE

                # Calculate new agent position
                agent.posX = agent.posX + (agent.v * math.cos(agent.angulo) * TIME_STEP)
                agent.posY = agent.posY + (agent.v * math.sin(agent.angulo) * TIME_STEP)
                agent.angulo = agent.angulo + (agent.w * TIME_STEP)

                # Check if agent is near the light in this step to add fitness
                if (distance(agent.posX, agent.posY, xLuz, yLuz) < DISTANCIA_MIN_FITNESS):
                    agent.stepsClose[luces] += 1

        # At the end of current light, save final distance
        fitnessValues[luces][1] = distance(centroidX(vAgents), centroidY(vAgents), xLuz, yLuz)

    # Rebuild the individual with the new weights
    indiv = []
    for agent in vAgents:
        vBias = [(i - BIAS_MIN)/(BIAS_MAX - BIAS_MIN) for i in agent.vBias]
        vTau = [(i - TAU_MIN)/(TAU_MAX - TAU_MIN) for i in agent.vTau]
        gainMotor = (agent.gainMotor - GAIN_MIN)/(GAIN_MAX - GAIN_MIN)
        gainSensor = (agent.gainSensor - GAIN_MIN)/(GAIN_MAX - GAIN_MIN)
        flatW = [item for sublist in agent.vW for item in sublist]
        flatP = [item for sublist in agent.vP for item in sublist]
        flatW = [(i - W_MIN)/(W_MAX - W_MIN) for i in agent.flatW]
        flatP = [(i - PLASTICIY_RATE_MIN)/(PLASTICIY_RATE_MAX - PLASTICIY_RATE_MIN) for i in agent.flatP]
        flatW = [item for sublist in agent.vW for item in sublist]
        flatP = [item for sublist in agent.vP for item in sublist]
        indiv.extend(vBias + vTau + [gainMotor] + [gainSensor] + flatW + flatP + agent.vPType)

    individual = indiv[:]

    # When all lights have been run, calculate fitness of the group of individuals
    finalFitness = []
    for i in range(0, N_LUCES):
        if (fitnessValues[i][1] > fitnessValues[i][0]):
            Fd = 0.0
        else:
            Fd = 1 - (fitnessValues[i][1] / fitnessValues[i][0])

        Fp = 0.0
        for agent in vAgents:
            Fp += agent.stepsClose[i] / fitnessValues[i][2]

        Fp = Fp / N_AGENTES

        Fh = 0.0
        for agent in vAgents:
            homeostaticas = 0
            for j in range(0, N_NEURONAS):
                if (agent.homeostatic[i][j] == 1):
                    homeostaticas += 1
            Fh += homeostaticas / N_NEURONAS

        Fh = Fh / N_AGENTES

        finalFitness.append((Fd * 0.34) + (Fp * 0.54) + (Fh * 0.12))


    return (sum(finalFitness) / N_LUCES),


#############################################################################################################
#################################### Global parameters and constants ########################################
N_NEURONAS_INTERMEDIAS = 2
N_NEURONAS = N_NEURONAS_INTERMEDIAS + 4
INDIVIDUAL_SIZE = 3 * N_NEURONAS + 2 + 2 * (N_NEURONAS * N_NEURONAS)
N_LUCES = 6
N_AGENTES = 4

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
DISTANCIA_INICIAL_AGENTE_MIN = RADIO_AGENTE * 4
DISTANCIA_INICIAL_AGENTE_MAX = RADIO_AGENTE * 8
T = 800 # Variable to calculate the time the lightsource will be on

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
    while(max(fits) < 0.89):
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
