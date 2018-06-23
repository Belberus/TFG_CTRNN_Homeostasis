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
    def __init__(self, identificador, posX, posY, angulo, bias, tau, motorGain, sensorGain, vW, vP, vPType):
        self.id = copy.deepcopy(identificador)
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
        self.alimentado = False

        self.v = 0.0
        self.w = 0.0

#############################################################################################################
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

# Function that checks if all the agents in the experiment have eaten from the light
def everyOneHasEaten(vector):
    for agent in vector:
        if (agent.alimentado == False):
            return False
    return True

#############################################################################################################
#################################### Global parameters and constants ########################################
N_NEURONAS_INTERMEDIAS = 2
N_NEURONAS = N_NEURONAS_INTERMEDIAS + 4
INDIVIDUAL_SIZE = 3 * N_NEURONAS + 2 + 2 * (N_NEURONAS * N_NEURONAS)
N_LUCES = 6
N_AGENTES = 5

CICLOS_PREVIOS = 50

RADIO_AGENTE = 4    # Agent radius
DIAMETRO_AGENTE = RADIO_AGENTE * 2  # Agent diameter
DISTANCIA_LUZ_MIN = RADIO_AGENTE * 10  # Min. distance that the lightsource can appear from the (0,0)
DISTANCIA_LUZ_MAX = RADIO_AGENTE * 25  # Max. distance that the lightsource can appear from the (0,0)
DISTANCIA_MIN_FITNESS = RADIO_AGENTE * 4    # Distance in wich we consider the agent close enough to the lightsource
INTENSIDAD_LUZ_MIN = 500 # Min. value of light intensity
INTENSIDAD_LUZ_MAX = 1500 # Max. value of light intensity
INTENSIDAD_VISUAL_AGENTE = 750
SEPARACIONSENSOR = 1.0472 # Separation between sensor position and agent axis angle, 60º
VISIONSENSOR = 1.39626 # Reading arc of the sensor in wich it reads light inputs, 80
DISTANCIA_INICIAL_AGENTE_MIN = RADIO_AGENTE * 4
DISTANCIA_INICIAL_AGENTE_MAX = RADIO_AGENTE * 8
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
##################################### Individual to test ####################################################

individual = []

#############################################################################################################
##################################### MAIN ##################################################################

if __name__ == "__main__":
    # Scale normalized [0,1] values to real values
    vAgents = []
    for agentID in range(0, N_AGENTES):
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
        vAgents.append(Agent(agentID, 0.0 + (math.cos(randomAngle) * randomDistance), 0.0 + (math.sin(randomAngle) * randomDistance), random.random() * (2 * math.pi), vBias, vTau, gainMotor, gainSensor, vW, vP, vPType))

    # Preparation of fitness array to calculate centroid fitness (each agent takes care of his own fitness)
    fitnessValues = []
    for i in range(0,N_LUCES):
        fitnessValues.append([0.0, 0.0])   # Colective puntuation, time the light has been ON

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

        # Save the time the light will be on
        fitnessValues[luces][1] = time

        # Auxiliar array to control the number of agents that are close to the light at the same time
        agentesCerca = [0 for _ in range(0,N_AGENTES)]

        # Once PreIntegration is finished, we can start our run
        for ciclos in range(0, time):
            for agent in vAgents:
                agent.inputs[0] = 0.0
                agent.inputs[1] = 0.0
                agent.inputs[2] = 0.0
                agent.inputs[3] = 0.0

                # Sensor 1 position
                rad1 = normalize(agent.angulo + SEPARACIONSENSOR)
                xSensor1 = agent.posX + ((RADIO_AGENTE) * math.cos(rad1))
                ySensor1 = agent.posY + ((RADIO_AGENTE) * math.sin(rad1))

                # Sensor 2 position
                rad2 = normalize(agent.angulo - SEPARACIONSENSOR)
                xSensor2 = agent.posX + ((RADIO_AGENTE) * math.cos(rad2))
                ySensor2 = agent.posY + ((RADIO_AGENTE) * math.sin(rad2))

            # First we update light sensors
                # Angle between light and agent
                angAgenteLuz = normalize(math.atan2(yLuz - agent.posY, xLuz - agent.posX) - agent.angulo)

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

            # Then we check agent visual sensors
                for agentToCheck in vAgents:
                    if agentToCheck.id != agent.id:
                        # Angle between the agent and the agent to check
                        angAgentCheck = normalize(math.atan2(agentToCheck.posY - agent.posY, agentToCheck.posX - agent.posX) - agent.angulo)
                        # Check if the agent sensor will be ON and update inputs
                        if (angAgentCheck <= llimit1):
                            ds1 = distance(xSensor1, ySensor1, agentToCheck.posX, agentToCheck.posY)**2
                            da = distance(agent.posX, agent.posY, agentToCheck.posX, agentToCheck.posY)
                            a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds1) / (da * da)
                            if (a <= 1.0):
                                agent.inputs[2] += INTENSIDAD_VISUAL_AGENTE / ds1
                            if(angAgentCheck <= llimit2):
                                ds2 = distance(xSensor2, ySensor2, agentToCheck.posX, agentToCheck.posY)**2
                                da = distance(agent.posX, agent.posY, agentToCheck.posX, agentToCheck.posY)
                                a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds2) / (da * da)
                                if (a<= 1.0):
                                    agent.inputs[3] += INTENSIDAD_VISUAL_AGENTE / ds2
                        elif (angAgentCheck >= hlimit2):
                            ds2 = distance(xSensor2, ySensor2, agentToCheck.posX, agentToCheck.posY)**2
                            da = distance(agent.posX, agent.posY, agentToCheck.posX, agentToCheck.posY)
                            a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds2) / (da * da)
                            if (a<= 1.0):
                                agent.inputs[3] += INTENSIDAD_VISUAL_AGENTE / ds2
                            if (angAgentCheck >= hlimit1):
                                ds1 = distance(xSensor1, ySensor1, agentToCheck.posX, agentToCheck.posY)**2
                                da = distance(agent.posX, agent.posY, agentToCheck.posX, agentToCheck.posY)
                                a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds1) / (da * da)
                                if (a <= 1.0):
                                    agent.inputs[2] += INTENSIDAD_VISUAL_AGENTE / ds1

                # Multiply with the gain
                agent.inputs[0] = agent.inputs[0] * agent.gainSensor
                agent.inputs[1] = agent.inputs[1] * agent.gainSensor
                agent.inputs[2] = agent.inputs[2] * agent.gainSensor
                agent.inputs[3] = agent.inputs[3] * agent.gainSensor

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

                # Check if agent is near the light in this step to check the collective puntuation
                if (distance(agent.posX, agent.posY, xLuz, yLuz) < DISTANCIA_MIN_FITNESS):
                    agentesCerca[agent.id] = 1
                    if (sum(agentesCerca) > 3):
                        fitnessValues[luces][0] -= 1
                    else:
                        if (agent.alimentado == False):
                            agent.alimentado = True
                            fitnessValues[luces][0] += 1
                        elif (agent.alimentado == True and everyOneHasEaten(vAgents) == False):
                            fitnessValues[luces][0] -= 1
                        elif (agent.alimentado == True and everyOneHasEaten(vAgents) == True):
                            fitnessValues[luces][0] += 1

                # If everyone has eaten, change everyone to not eaten
                if (everyOneHasEaten(vAgents) == True):
                    for a in vAgents:
                        a.alimentado = False

        # At the end of current light, we do nothing anymore
        # fitnessValues[luces][1] = distance(centroidX(vAgents), centroidY(vAgents), xLuz, yLuz)

    # Rebuild the individual with the new weights
    indiv = []
    for agent in vAgents:
        vBias = [(i - BIAS_MIN)/(BIAS_MAX - BIAS_MIN) for i in agent.vBias]
        vTau = [(i - TAU_MIN)/(TAU_MAX - TAU_MIN) for i in agent.vTau]
        gainMotor = (agent.gainMotor - GAIN_MIN)/(GAIN_MAX - GAIN_MIN)
        gainSensor = (agent.gainSensor - GAIN_MIN)/(GAIN_MAX - GAIN_MIN)
        flatW = [item for sublist in agent.vW for item in sublist]
        flatP = [item for sublist in agent.vP for item in sublist]
        flatW = [(i - W_MIN)/(W_MAX - W_MIN) for i in flatW]
        flatP = [(i - PLASTICIY_RATE_MIN)/(PLASTICIY_RATE_MAX - PLASTICIY_RATE_MIN) for i in flatP]
        flatW = [item for sublist in agent.vW for item in sublist]
        flatP = [item for sublist in agent.vP for item in sublist]
        indiv.extend(vBias + vTau + [gainMotor] + [gainSensor] + flatW + flatP + agent.vPType)

    individual = indiv[:]

    # When all lights have been run, calculate fitness of the group of individuals
    # Sum all the puntuation in the lights and the total time of the lights
    puntuacion = 0.0
    totalTime = 0.0
    for i in range(0, N_LUCES):
        puntuacion += fitnessValues[i][0]
        totalTime += fitnessValues[i][1]

    Fd =  puntuacion / (totalTime * 3)

    FhVector = []
    for i in range(0, N_LUCES):
        FhTemp = 0.0
        for agent in vAgents:
            homeostaticas = 0
            for j in range(0, N_NEURONAS):
                if (agent.homeostatic[i][j] == 1):
                    homeostaticas += 1
            FhTemp += homeostaticas / N_NEURONAS

        FhTemp = FhTemp / N_AGENTES
        FhVector.append(FhTemp)

    Fh = sum(FhVector) / N_LUCES

    return ((Fd * 0.88) + (Fh * 0.12)),
