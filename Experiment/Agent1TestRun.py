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

NOISE_SENSOR = False
NOISE_DEVIATION = 0.44

NOISE_HOMEOSTATIC = False
NOISE_WEIGHT = True

#############################################################################################################
##################################### Individuo 1 para probar ###############################################

individual = [0.6265855494470844, 0.5336029415372073, 0.46910422201753965, 0.3475924752151335, 0.14174634037268075, 0.6447004717886272, 0.027597598192843797, 0.7753142285318443, 0.1133206864802816, 0.16196151269492032, 0.4725831658473808, 0.5688407893538755, 0.46006102943835625, 0.6699096109867527, 0.779398327450121, 0.887558443401648, 0.49323197884168557, 0.8705770162541511, 0.6426528687067358, 0.6653766737163902, 0.14904019351290654, 0.3193177876092731, 0.4860345826594933, 0.5590926271360919, 0.6780809955266598, 0.7700957276711914, 0.4543007545103567, 0.24632511678777835, 0.3176916178786702, 0.14535622125327696, 0.8269691901648206, 0.9802472418676097, 0.13074852472654297, 0.8705324243000392, 0.23331938029034172, 0.24970765368907, 0.8765776462616929, 0.2756258997244443, 0.11961466825675682, 0.16946346712321192, 0.22398791890265712, 0.6750522488432801, 0.1886256534669718, 0.043623744891451266, 0.26190760980162175, 0.09397874154277785, 0.9787310801110788, 0.07569403859625512, 0.19659940916683039, 0.330079424170141, 0.8064803762188072, 0.41282788487852407, 0.46633635616629165, 0.37661979892431197, 0.13870172557284366, 0.07754896818719326, 0.9280561177249538, 0.6040593394599361, 0.5200844001288576, 0.7263221264247316, 0.40517697795046825, 0.8738126871613897, 0.24381709155910525, 0.9291574109494856, 0.08518373034353244, 0.595871996924665, 0.9946015831828385, 0.047768386436019994, 0.40007897373349643, 0.7297195546404509, 0.47142667828821205, 0.7948727771233551, 0.4735690893619132, 0.6288513550383776, 0.9367738275198084, 0.019892586178180016, 0.3661995172873379, 0.5255600912527496, 0.9232247459905462, 0.6609859441100074, 0.9784095206566034, 0.2823685600761213, 0.06537271206397965, 0.10504430744694293, 0.8367366493069451, 0.5970065295807457, 2, 0, 3, 3, 0, 3, 0.3046160487305324, 0.010281273008059921, 0.8104260046529114, 0.5767363425455935, 0.8062262563084764, 0.35636170306536485, 0.349767206060815, 0.4711772609086048, 0.5194669314287816, 0.3027597197568963, 0.17907229978141204, 0.6969819648159425, 0.39839147976431677, 0.07934161550409846, 0.35025239986366385, 0.6683432756705868, 0.25520294237988084, 0.5397009652456521, 0.9724457834009962, 0.6878770358123233, 0.10568916143056717, 0.994983076726823, 0.2580974190683072, 0.012277180648283537, 0.9165450759670388, 0.4318760979036673, 0.4321545311517482, 0.6355203147702511, 0.5805592435696849, 0.15388976515542396, 0.6770318894016312, 0.3665827274370773, 0.22385999552777147, 0.9617976887474604, 0.6740888722105745, 0.43838905844855847, 0.11384190386850235, 0.02811606460683247, 0.7038246973254555, 0.032447678738439234, 0.6151826866675582, 0.7476500522724401, 0.9922254582998091, 0.5709723060782523, 0.5237322032027968, 0.34358214956910615, 0.4100416650459652, 0.4788143402771269, 0.24733925893915154, 0.08395986098195352, 0.846356584280722, 0.5583033316690542, 0.8879206844308386, 0.9181956890508897, 0.745020919243765, 0.4589675919637898, 0.7761241258691216, 0.8808108257107962, 0.14858746257927158, 0.9399558234885222, 0.5334964038382166, 0.5573623013159309, 0.010669625379727155, 0.8675899139713987, 0.05961646456610192, 0.19359229029588376, 0.24788877835670498, 0.9528065789267927, 0.6054630555366775, 0.5587038499756909, 0.24820584925266542, 0.41455525859070685, 0.3874939174832661, 0.030213859409450095, 0.6169165970868259, 0.44868605396170325, 0.013149550627649553, 0.8903154717437316, 0.5364000046070522, 0.6994758166906841, 0.12168464230974563, 0.27588369135796176, 0.10691900977944002, 0.5166209968907258, 0.2824246942431945, 0.5811571200366571, 3, 3, 1, 2, 2, 2, 0.9728854303448387, 0.4130973104192276, 0.31648926770346975, 0.43811260679990816, 0.33537425703370916, 0.10680932275720412, 0.40682336619407944, 0.5847131697422483, 0.6356967547748408, 0.2637348350837553, 0.9293357487337768, 0.7049837496410483, 0.9104632581475601, 0.3896565331605316, 0.0803922427015552, 0.40498466830968605, 0.4305513038325339, 0.10864113433001144, 0.5540557860828088, 0.0009884467354414062, 0.32623391243154054, 0.1644875732185418, 0.24368260345310722, 0.9856784673681696, 0.29549741245423056, 0.5118728741691336, 0.08323605906565712, 0.7082958130735417, 0.9914942702091888, 0.6170717902370769, 0.23833944368245308, 0.20723326079659588, 0.789725042886876, 0.09276511457694259, 0.3449302728699223, 0.38230432848794527, 0.7206732732531773, 0.039420070025179665, 0.8102709264798353, 0.7858694158481555, 0.0732874905009182, 0.7537069613363295, 0.28342616019488487, 0.448620479654678, 0.45814822732614924, 0.31974993945002683, 0.24115141209511826, 0.08194454143210261, 0.8493834783144248, 0.5615216001085115, 0.8914685786471688, 0.6061925570954069, 0.23795612461829996, 0.10433610985236863, 0.649433590843704, 0.5931284344109653, 0.6157839641040354, 0.3358310345332697, 0.946027187523912, 0.6301497391266583, 0.6899603426951779, 0.19935418503637503, 0.2291190948372821, 0.4028283234565909, 0.4019013531614184, 0.3516934250045747, 0.33445383062730927, 0.25317780985721483, 0.533465741145574, 0.3130089108978237, 0.5856874785760483, 0.19177898726410114, 0.48320148853471656, 0.8019859440998615, 0.14940265554444232, 0.2167548195264809, 0.4337561100644698, 0.3608823517683598, 0.9643842200539493, 0.5649552053657535, 0.08780473853820658, 0.5014988133423715, 0.6347706528949272, 0.8349259395955287, 0.5617558448867599, 0.8759942605626292, 2, 0, 1, 3, 0, 1, 0.2954073676258455, 0.1002972549744281, 0.1276672423954459, 0.7269053528473451, 0.21210630374210993, 0.4574583135228554, 0.38217845174321485, 0.506642809409686, 0.293216849778727, 0.7792240061153053, 0.47083102759959317, 0.5327497634767765, 0.7623488543260942, 0.6270832080125183, 0.8499381332181052, 0.3702594378281533, 0.5334858768469114, 0.8133107849405418, 0.11124002274979694, 0.0922904343357357, 0.6750968028384228, 0.04270890372522729, 0.9704641827062527, 0.9331944112693541, 0.7149729309516265, 0.7841399627700799, 0.3012472818207088, 0.7248667194986659, 0.19045128768829367, 0.35877458300011633, 0.7862138092109805, 0.13480349827166116, 0.375889236915642, 0.1719542085152812, 0.15058634994286357, 0.15372862820793465, 0.9246420277610103, 0.9483338591138691, 0.877735487402083, 0.9257601095586647, 0.10494681678848894, 0.08334079476597167, 0.5546463001408877, 0.06983654419852525, 0.2376299797062137, 0.7258784234308491, 0.6687173129020016, 0.8197680418520799, 0.852725369883318, 0.2827499339531996, 0.8764379624886689, 0.43543691714586896, 0.07955459959615385, 0.6773043873268162, 0.2400479610448344, 0.29424960625421204, 0.6432639422874239, 0.2076056049435352, 0.5103360937935987, 0.16669309512976305, 0.9008795830965987, 0.4441831686147839, 0.7973695326856016, 0.3836621063283714, 0.508871269170629, 0.9711573092190853, 0.5989325744129054, 0.4970423649341499, 0.44276225871603603, 0.2339847191981479, 0.5573606763175631, 0.9705046797226442, 0.0011293597047644655, 0.7409109398822168, 0.9725440087100902, 0.6492619584096418, 0.4587919962136092, 0.6996619011822469, 0.8988259043953544, 0.5612073346543807, 0.7042521663942829, 0.13963289392314282, 0.8460873835018743, 0.6662272894045422, 0.9434131696597181, 0.47745040254995574, 3, 1, 2, 0, 0, 0, 0.8202636612316768, 0.2508733655171208, 0.08252521424099146, 0.9982207922005033, 0.7017565313012076, 0.45957166136913596, 0.21587907970091003, 0.9307151574870888, 0.3641411791566128, 0.48758609775433837, 0.20623767428137119, 0.4064644072271818, 0.03016551096452713, 0.739300671514497, 0.6666831879292138, 0.39390814183549117, 0.9806630534439136, 0.18828151405760796, 0.5737033593270974, 0.030547046056969118, 0.9508101266553937, 0.31776711016001224, 0.8107754352143198, 0.2583425849117402, 0.8990128234316839, 0.5034401985657663, 0.0920955017485795, 0.3491225890456101, 0.5958051584783957, 0.8189784457443677, 0.7562200146630061, 0.03855115455568414, 0.6731975316043742, 0.1370233522242008, 0.06336993966002935, 0.4887278016253216, 0.6885399785926645, 0.9704354199779349, 0.08903004611035759, 0.5238257229391562, 0.026968117783984447, 0.7397819381008847, 0.27968108383410195, 0.012140989940887081, 0.4443378162382101, 0.6433662835794436, 0.4170612435441222, 0.18714304983796493, 0.971313649015661, 0.2684866869689724, 0.22535487962882905, 0.03752557472866225, 0.4868639249795714, 0.11164936985271268, 0.9905434109201771, 0.4068364192621232, 0.6090351662535622, 0.17313387309885386, 0.863361408650431, 0.9251771492394223, 0.5145537827179902, 0.8124187716765913, 0.38175338896036415, 0.001338441068794216, 0.0103689655813749, 0.5894942901080116, 0.18948821690539763, 0.7586659800717108, 0.6704117872710098, 0.8758098406230208, 0.9732189645286049, 0.9654658045186874, 0.18269060265154125, 0.446321839822436, 0.23398175915441355, 0.4348817065100098, 0.14102576438566006, 0.7189122976799537, 0.6485463783074704, 0.9058911521543174, 0.5991179851201679, 0.2749938693180285, 0.4131028875568944, 0.505076369127934, 0.009430567211277996, 0.6854665174574653, 3, 2, 2, 3, 2, 0]



#############################################################################################################
##################################### MAIN ##################################################################

if __name__ == "__main__":
    # Preparation of arrays to save statistics about the run
    trayectoriasX = [[] for _ in range(0, N_AGENTES)]   # X positions
    trayectoriasY = [[] for _ in range(0, N_AGENTES)]   # Y positions

    lastPositionX = [[[] for _ in range(0, N_LUCES)] for _ in range(0, N_AGENTES)]   # X last positions
    lastPositionY = [[[] for _ in range(0, N_LUCES)] for _ in range(0, N_AGENTES)]   # Y last positions

    positionLuzX = []
    positionLuzY = []

    activaciones = [[[] for _ in range(0, N_NEURONAS)] for _ in range(0, N_AGENTES)]    # Activation of the neurons
    homeostaticos = [[[] for _ in range(0, N_NEURONAS)] for _ in range(0, N_AGENTES)]   # Homeostatic value of each neuron

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
    collectiveFitness = []
    for i in range(0,N_LUCES):
        collectiveFitness.append([0.0, 0.0])   # Colective puntuation, time the light has been ON

    individualFitness = []
    for i in range(0, N_AGENTES):
        individualFitness.append([[0.0, 0.0] for _ in range(0, N_LUCES)])   # Individual puntuation. Each agent, each light, initial distance and final distance

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

        positionLuzX.append(xLuz)
        positionLuzY.append(yLuz)

        # Save the time the light will be on
        collectiveFitness[luces][1] = time

        # Save the initial distance of each agent to that light
        for i in range(0, N_AGENTES):
            individualFitness[i][luces][0] = distance(vAgents[i].posX, vAgents[i].posY, xLuz, yLuz)

        # Auxiliar array to control the number of agents that are close to the light at the same time
        agentesCerca = [0 for _ in range(0,N_AGENTES)]

        # Once PreIntegration is finished, we can start our run
        for ciclos in range(0, time):
            for agent in vAgents:
                agent.inputs[0] = 0.0
                agent.inputs[1] = 0.0
                agent.inputs[4] = 0.0
                agent.inputs[5] = 0.0

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
                                agent.inputs[4] += INTENSIDAD_VISUAL_AGENTE / ds1
                            if(angAgentCheck <= llimit2):
                                ds2 = distance(xSensor2, ySensor2, agentToCheck.posX, agentToCheck.posY)**2
                                da = distance(agent.posX, agent.posY, agentToCheck.posX, agentToCheck.posY)
                                a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds2) / (da * da)
                                if (a<= 1.0):
                                    agent.inputs[5] += INTENSIDAD_VISUAL_AGENTE / ds2
                        elif (angAgentCheck >= hlimit2):
                            ds2 = distance(xSensor2, ySensor2, agentToCheck.posX, agentToCheck.posY)**2
                            da = distance(agent.posX, agent.posY, agentToCheck.posX, agentToCheck.posY)
                            a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds2) / (da * da)
                            if (a<= 1.0):
                                agent.inputs[5] += INTENSIDAD_VISUAL_AGENTE / ds2
                            if (angAgentCheck >= hlimit1):
                                ds1 = distance(xSensor1, ySensor1, agentToCheck.posX, agentToCheck.posY)**2
                                da = distance(agent.posX, agent.posY, agentToCheck.posX, agentToCheck.posY)
                                a = (((RADIO_AGENTE) * (RADIO_AGENTE)) + ds1) / (da * da)
                                if (a <= 1.0):
                                    agent.inputs[4] += INTENSIDAD_VISUAL_AGENTE / ds1

                if (NOISE_SENSOR == False):
                    # Multiply with the gain
                    agent.inputs[0] = agent.inputs[0] * agent.gainSensor
                    agent.inputs[1] = agent.inputs[1] * agent.gainSensor
                    agent.inputs[4] = agent.inputs[4] * agent.gainSensor
                    agent.inputs[5] = agent.inputs[5] * agent.gainSensor
                else:
                    ruido = agent.inputs[0] * NOISE_DEVIATION
                    agent.inputs[0] = random.uniform(agent.inputs[0] - ruido, agent.inputs[0] + ruido) * agent.gainSensor
                    ruido = agent.inputs[1] * NOISE_DEVIATION
                    agent.inputs[1] = random.uniform(agent.inputs[1] - ruido, agent.inputs[1] + ruido) * agent.gainSensor
                    ruido = agent.inputs[4] * NOISE_DEVIATION
                    agent.inputs[4] = random.uniform(agent.inputs[4] - ruido, agent.inputs[4] + ruido) * agent.gainSensor
                    ruido = agent.inputs[5] * NOISE_DEVIATION
                    agent.inputs[5] = random.uniform(agent.inputs[5] - ruido, agent.inputs[5] + ruido) * agent.gainSensor

                # Make CTRNN RUN
                for i in range(0, N_NEURONAS - 2):
                    change = -agent.outputs[i]

                    for j in range(0, N_NEURONAS - 2):
                        temp = agent.outputs[j] + agent.vBias[j]
                        change += agent.vW[j][i] * sigmoid(temp)

                    change = change + agent.inputs[i]
                    change = change / agent.vTau[i]

                    agent.outputs[i] = agent.outputs[i] + (change * TIME_STEP)
                    activaciones[agent.id][i].append(agent.outputs[i])

                for i in range(N_NEURONAS - 2, N_NEURONAS):
                    change = -agent.outputs[i]

                    for j in range(N_NEURONAS - 2, N_NEURONAS):
                        temp = agent.outputs[j] + agent.vBias[j]
                        change += agent.vW[j][i] * sigmoid(temp)

                    change = change + agent.inputs[i]
                    change = change / agent.vTau[i]

                    agent.outputs[i] = agent.outputs[i] + (change * TIME_STEP)
                    activaciones[agent.id][i].append(agent.outputs[i])

                # Allow plasticity changes in motor network
                for i in range(0, N_NEURONAS - 2):
                    for j in range(0, N_NEURONAS - 2):
                        if agent.vPType != 0:   # If there is no plasticity, nothing is done
                            weight = agent.vW[i][j]
                            jY = agent.outputs[j]
                            jBias = agent.vBias[j]
                            iN = agent.vP[i][j]
                            iRate = sigmoid(agent.outputs[i] + agent.vBias[i])
                            jRate = sigmoid(agent.outputs[j] + agent.vBias[j])

                            if (NOISE_HOMEOSTATIC == True):
                                ruido = iN * NOISE_DEVIATION
                                iN = random.uniform(iN - ruido, iN + ruido)

                            if (NOISE_WEIGHT == True):
                                ruido = weight * NOISE_DEVIATION
                                weight = random.uniform(weight - ruido, weight + ruido)

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

                # Allow plasticity changes in collective sensor network
                for i in range(N_NEURONAS - 2, N_NEURONAS):
                    for j in range(N_NEURONAS - 2, N_NEURONAS):
                        if agent.vPType != 0:   # If there is no plasticity, nothing is done
                            weight = agent.vW[i][j]
                            jY = agent.outputs[j]
                            jBias = agent.vBias[j]
                            iN = agent.vP[i][j]
                            iRate = sigmoid(agent.outputs[i] + agent.vBias[i])
                            jRate = sigmoid(agent.outputs[j] + agent.vBias[j])

                            if (NOISE_HOMEOSTATIC == True):
                                ruido = iN * NOISE_DEVIATION
                                iN = random.uniform(iN - ruido, iN + ruido)

                            if (NOISE_WEIGHT == True):
                                ruido = weight * NOISE_DEVIATION
                                weight = random.uniform(weight - ruido, weight + ruido)

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
                vr = sigmoid(agent.outputs[2] + agent.vBias[2] + agent.outputs[4] + agent.vBias[4])
                vl = sigmoid(agent.outputs[3] + agent.vBias[3] + agent.outputs[5] + agent.vBias[5])

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

                # Append agent position in this step
                trayectoriasX[agent.id].append(agent.posX)
                trayectoriasY[agent.id].append(agent.posY)

                # Check if agent is near the light in this step to check the collective puntuation
                if (distance(agent.posX, agent.posY, xLuz, yLuz) < DISTANCIA_MIN_FITNESS):
                    agentesCerca[agent.id] = 1
                    if (sum(agentesCerca) > 3):
                        collectiveFitness[luces][0] -= (1 * N_AGENTES)
                    else:
                        collectiveFitness[luces][0] += 1
                else:
                    agentesCerca[agent.id] = 0
        # At the end of the light, calculate the final distance of each agent to the light
        for i in range(0, N_AGENTES):
            individualFitness[i][luces][1] = distance(vAgents[i].posX, vAgents[i].posY, xLuz, yLuz)

        lastPositionX[agent.id][luces] = agent.posX
        lastPositionY[agent.id][luces] = agent.posY

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

    temporalValues = [0.0 for _ in range(0, N_AGENTES)]
    Fd = 0.0
    FdTemp = 0.0
    for i in range(0, N_AGENTES):
        for j in range(0, N_LUCES):
            if (individualFitness[i][j][1] > individualFitness[i][j][0]):
                FdTemp = 0.0
            else:
                FdTemp = 1 - (individualFitness[i][j][1] / individualFitness[i][j][0])
            temporalValues[i] += FdTemp

        temporalValues[i] = temporalValues[i] / N_LUCES

    Fd = sum(temporalValues) / N_AGENTES

    puntuacion = 0.0
    totalTime = 0.0
    for i in range(0, N_LUCES):
        puntuacion += collectiveFitness[i][0]
        totalTime += collectiveFitness[i][1]

    Fp =  puntuacion / (totalTime * 3)

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

    totalFitness = ((Fd * 0.44) + (Fp * 0.44) + (Fh * 0.12)),

    print(totalFitness)

    # # Print the results of the execution
    # plt.scatter(positionLuzX, positionLuzY, s=60, c='red', marker='^')
    # for i in range(0, len(positionLuzX)):
    #     plt.annotate(i, (positionLuzX[i],positionLuzY[i]))
    #
    #
    # plt.plot(trayectoriasX[0][0], trayectoriasY[0][0], "bx")
    # plt.plot(trayectoriasX[1][0], trayectoriasY[1][0], "rx")
    # plt.plot(trayectoriasX[2][0], trayectoriasY[2][0], "gx")
    # plt.plot(trayectoriasX[3][0], trayectoriasY[3][0], "cx")
    # plt.plot(trayectoriasX[4][0], trayectoriasY[4][0], "kx")
    # plt.plot(trayectoriasX[0], trayectoriasY[0], "b-")
    # plt.plot(trayectoriasX[1], trayectoriasY[1], "r-")
    # plt.plot(trayectoriasX[2], trayectoriasY[2], "g-")
    # plt.plot(trayectoriasX[3], trayectoriasY[3], "c-")
    # plt.plot(trayectoriasX[4], trayectoriasY[4], "k-")
    # plt.show()
    #
    # for i in range(0, N_AGENTES):
    #     file = open("Results1/activaciones" + str(i) + ".txt", "w")
    #     for j in range(0, N_NEURONAS):
    #         file.write(str(activaciones[i][j]) + "\n")
    #         file.write("---------------------------------------------- \n")
    #     file.close()
    #
    # # Save light positions
    # file = open("Results1/luces.txt", "w")
    # for i in range(0, N_LUCES):
    #     file.write(str(positionLuzX[i]) + " - " + str(positionLuzY[i]) + "\n")
    # file.close()
    #
    # for i in range(0, N_AGENTES):
    #     file = open("Results1/trayectorias" + str(i) + ".txt", "w")
    #     file.write(str(trayectoriasX[i]) + "\n")
    #     file.write("---------------------------------------------- \n")
    #     file.write(str(trayectoriasY[i]) + "\n")
    #     file.close()
