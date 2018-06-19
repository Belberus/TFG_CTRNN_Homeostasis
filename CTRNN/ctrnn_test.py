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

T = 2400 # Variable to calculate the time the lightsource will be on

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

individual = [0.01746243308444806, 0.5161066079927694, 0.5436766074450864, 0.11067892355463305, 0.3425258914777899, 0.28717684649157593, 0.09033945916006436, 0.7209209852810927, 0.6645600274845561, 0.5714785510797074, 0.22572250646516911, 0.422611807744954, 0.18916816124718194, 0.6205121813593082, 0.4312550497806289, 0.6031470029180175, 0.4384873234381108, 0.73565371845403, 0.8393702812084912, 0.06546638212461242, 0.3613336777166154, 0.3851179176895224, 0.22602106359007257, 0.18558845999352858, 0.0621537285883742, 0.6103552147230041, 0.031199813780289576, 0.8298455327981075, 0.7263777751767857, 0.821054429198789, 0.6799934286771334, 0.034924717837380825, 0.6459383433487249, 0.5073096974859929, 0.18662534346896786, 0.6188417738360388, 0.9893146401674413, 0.8221859975495303, 0.9651955531015373, 0.22780292862618923, 0.7454748147705599, 0.37992009194363274, 2, 2, 1, 3]


# individual = [0.11745319602657645, 0.4324368963070979, 0.298646401950595, 0.527648680654615, 0.705087443435338, 0.9857224041723999, 0.41901589682927864, 0.9427379809271011, 0.048629153108834644, 0.7389329151023168, 0.42067604774254075, 0.592997575146583, 0.1021967120613696, 0.704488556513817, 0.023654741142900337, 0.6621500238907855, 0.7660566697353497, 0.9688765522766395, 0.5721630537737386, 0.1351780458376678, 0.17351535367865256, 0.06529793102909698, 0.622448921050077, 0.010302659747702947, 0.5744213821655519, 0.013016752907966334, 0.9951903492841252, 0.9004829806400144, 0.6087311961141251, 0.45777309147638856, 0.6125134701596482, 0.09028325524885461, 0.8035329329647045, 0.45448776997482787, 0.8361127812280127, 0.3132361074607578, 0.8267491989023095, 0.4226156410942977, 0.48842691847827124, 0.644183336289239, 0.9900976001328811, 0.30200157303354747, 2, 3, 0, 3]


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
