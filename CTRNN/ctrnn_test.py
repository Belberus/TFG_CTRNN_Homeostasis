import random
from random import randint
import matplotlib.pyplot as plt
import math


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

# Place here the individual to execute

individual = [0.4637589817483826, 0.5206593015892258, 0.22037883033426264, 0.19186558683627586, 0.7747557186550458, 0.9718863875406384, 0.5634832797905601, 0.016400420826465534, 0.1773000352960914, 0.9413669581925369, 0.37594366909245824, 0.6060078690140903, 0.8191632545648467, 0.3993969937235108, 0.22681795553945783, 0.18304953298267324, 0.4866098018386218, 0.11529458605492837, 0.11157992465322553, 0.46602538118138526, 0.323425495927659, 0.4711903664702538, 0.6237118327543344, 0.602239342829039, 0.2190665822361212, 0.8712786112355936, 0.7450862757996197, 0.9595880548632582, 0.9021063554237656, 0.14649943538174937, 0.9855319961956532, 0.8595574547853649, 0.6346672696971759, 9.358686097959978e-05, 0.467177282174457, 0.6477962831790348, 0.06359712353151281, 0.4000436030342709, 0.8057938305059938, 0.8100309524360734, 0.5688355167557202, 0.9671898957197704, 1, 3, 0, 2]



#############################################################################################################
##################################### MAIN ##################################################################

if __name__ == "__main__":
    # Representation variables
    historialX = []
    historialY = []
    lucesX = []
    lucesY = []
    endAgenteX = []
    endAgenteY = []

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

        lucesX.append(xLuz)
        lucesY.append(yLuz)

        # PreIntegration in order to let the individual stabilize
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

            # Check if the sensors will be ON and update inputs
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

        # Once PreIntegration is finished, we can start our run
        for ciclos in range(0, time):
            historialX.append(xAgente)
            historialY.append(yAgente)

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

            # Check if the sensors will be ON and update inputs
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

        # At the end of current light, save the agent positon
        endAgenteX.append(xAgente)
        endAgenteY.append(yAgente)

    # Print the results of the execution
    plt.scatter(lucesX, lucesY, s=60, c='red', marker='^')
    plt.scatter(endAgenteX, endAgenteY, s=30, c='blue', marker='o')
    for i in range(0, len(lucesX)):
        plt.annotate(i, (lucesX[i],lucesY[i]))
        plt.annotate(i, (endAgenteX[i],endAgenteY[i]))
    plt.plot(historialX[0], historialY[0], "bo")
    plt.plot(historialX, historialY)
    plt.show()
