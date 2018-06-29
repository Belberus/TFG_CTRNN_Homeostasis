import random
from random import randint
import numpy as np
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

NOISE_SENSOR = False
NOISE_DEVIATION = 0.44

NOISE_HOMEOSTATIC = False
NOISE_WEIGHT = True

#############################################################################################################
#################################### Individual (Type 2) to test ############################################

individual = [0.48824602670755, 0.23121810393632847, 0.3440582903579358, 0.9121566158661613, 0.33062129108589833, 0.13421858264326425, 0.529513404835132, 0.2858363263891913, 0.7274674728185658, 0.07482145568679666, 0.4785177753851153, 0.34012819610425205, 0.38959349936005605, 0.906285567720605, 0.4301962518941115, 0.5406439968892774, 0.07077673438286447, 0.2158439054813377, 0.0860789477969991, 0.6600807027709313, 0.3539798298241068, 0.0820018096384586, 0.6160687615577441, 0.949267919701782, 0.8127467861878267, 0.6507717591394622, 0.38069299611803586, 0.22317237551630797, 0.12986896952185323, 0.32445857906853093, 0.5610063881207867, 0.5740498388819164, 0.35719300131293485, 0.07499636998387937, 0.47759705060678226, 0.06251395171295993, 0.22059690696607803, 0.711262236985129, 0.1346976562447424, 0.20647490614511255, 0.5470958016767746, 0.45025320160420057, 0.2972619850149588, 0.32023051862176044, 0.5468563773213923, 0.46294750196695567, 0.09039222196014074, 0.5272968158571836, 0.5448313064652632, 0.02948483295713844, 0.651226826999938, 0.7764472995453223, 0.973257149598036, 0.9757770067810048, 0.1827200947602694, 0.9884302242231183, 0.0674726223694263, 0.06205897427507556, 0.523376603180886, 0.10434278575814848, 0.8025527679865843, 0.38204657452580526, 0.9943655814972037, 0.9802990327745381, 0.9727907630227428, 0.3293473058214499, 0.6250303388638112, 0.9583684928694393, 0.6583825814907143, 0.22447244071800665, 0.9889563103631734, 0.8049770924753964, 0.1528169104105258, 0.945227737978252, 0.36290336779527643, 0.4644604660211621, 0.4783264137762723, 0.26689814854480354, 0.5992303704600127, 0.03882538549700498, 0.6976530264392911, 0.7060403166416733, 0.4737829899423176, 0.6594129587747011, 0.04679448345346937, 0.6322510155904196, 2, 1, 0, 0, 2, 1, 0.6448237633981231, 0.07273107776512999, 0.5162169993015152, 0.13359772059413744, 0.09463055988169089, 0.20399492453473622, 0.05096319827962159, 0.7534577574350891, 0.4376606605282748, 0.5935772732112782, 0.6128753128571328, 0.2461249097564827, 0.29210551900634807, 0.6142773170182874, 0.17167249716971855, 0.428244241729581, 0.5111656490224437, 0.22513380379271686, 0.059936989476678915, 0.0526672837143356, 0.21736187699726273, 0.8568071578190743, 0.9948339767992073, 0.7190647163254902, 0.8398601429156143, 0.8387439768441727, 0.78312572437906, 0.6784356122681704, 0.608114543319459, 0.27331694706041965, 0.8763367000066875, 0.2694790071588383, 0.016503933963770923, 0.19828916495365378, 0.7363930022457683, 0.6969098725439948, 0.4772947799947742, 0.21913565965365311, 0.24682728370861629, 0.6025739234783878, 0.2869102827030413, 0.5774264067846506, 0.2528904035073807, 0.8674684762617617, 0.4462827421226543, 0.4285462398186455, 0.6645319814656325, 0.49565888832199256, 0.5510542571725076, 0.4182460914989846, 0.2502581980702929, 0.9373836700552549, 0.2684627364146017, 0.6887329813935206, 0.24948594057171403, 0.9499542518368816, 0.008095957613321492, 0.8113067184602192, 0.8290568887948391, 0.6942077717358093, 0.46378342435011666, 0.9487941331854751, 0.5364082501333529, 0.08682197090975707, 0.15035842810458666, 0.2576527906246786, 0.9219965540928888, 0.5717037605509985, 0.8728593912872831, 0.8301081497102105, 0.6845751849597187, 0.014043146458909916, 0.022038693442744894, 0.45123314005620674, 0.2712162477625776, 0.833128994152827, 0.025051205915415875, 0.275062438248172, 0.45996922998455436, 0.3838465735178068, 0.8997347039472334, 0.38175694150440154, 0.020572371546493384, 0.9993373389122165, 0.013185678016070734, 0.5303981738279336, 2, 2, 1, 3, 1, 0, 0.579416370126571, 0.6024764196827299, 0.3259525636663839, 0.9211919995389054, 0.57544637696779, 0.4170979761686078, 0.6521255560138666, 0.3987518800115015, 0.4917058764633705, 0.22617465709997253, 0.23556778561519653, 0.7613769064702318, 0.3524179281692633, 0.3728700104060718, 0.3162979742196447, 0.21816777265578158, 0.002885953832394761, 0.7193655912720761, 0.5649918278226701, 0.4357733460579064, 0.2396201549322522, 0.3863937634880179, 0.35324066071666926, 0.327047890091993, 0.3829510736864794, 0.2061000745851833, 0.37708449695532764, 0.6647890879383653, 0.24297017423097478, 0.2642832142896091, 0.7960781528956666, 0.061432312880548956, 0.19862479735570504, 0.1008358296054872, 0.4432011460901635, 0.7858544520726386, 0.7457384908144848, 0.020382954252763152, 0.9099125312550801, 0.961020414635079, 0.7755345146419659, 0.5139121639739863, 0.625290238305986, 0.9792910571467888, 0.9169954273701757, 0.5077973174211476, 0.22011537299566142, 0.6104800511251005, 0.6125888518964647, 0.5477660797932511, 0.0842384625033552, 0.06473954853830655, 0.44515536464143657, 0.681340765725689, 0.24237194693671127, 0.9540985936223894, 0.9031333759715994, 0.785100779647601, 0.6822013676216232, 0.30590191771429687, 0.9995146442426543, 0.9659490725427639, 0.44768238452413134, 0.36032458555222124, 0.7905914942254638, 0.26109108207317444, 0.04958395595122378, 0.07209545800946016, 0.727536790199563, 0.4790005848153017, 0.09425129328854587, 0.4307729608940495, 0.06820686316043856, 0.20982821805346263, 0.8491250674301591, 0.5721247281013936, 0.28972407801033584, 0.2904102928467833, 0.16499315666466552, 0.4275991863116123, 0.3483503506558948, 0.543868544635414, 0.30313198481196924, 0.11781895102122042, 0.928968431634477, 0.32500691607326804, 2, 1, 2, 0, 0, 1, 0.30295130460496367, 0.6830350839559448, 0.6527789168062901, 0.8099983783148152, 0.7927968340547497, 0.9422096556185949, 0.2228411382635488, 0.1789821162935099, 0.8869741756498465, 0.7037971774612406, 0.29805654528810155, 0.7898480553796341, 0.6998468461998261, 0.6408505778714398, 0.573970034297586, 0.3523602939997782, 0.21202772513891022, 0.3961409841783361, 0.8186862528365454, 0.30449739171972445, 0.7393024466806647, 0.1527799336437904, 0.7379102120320054, 0.6764961771773971, 0.1997630879273954, 0.10381206231081075, 0.4056789698269796, 0.6051403747052353, 0.4504713484264745, 0.022120792030228653, 0.887421152249851, 0.22218238152288627, 0.6100012420854267, 0.31905084028094943, 0.8551223306400169, 0.07385356771732066, 0.7578612742433659, 0.04401972573945001, 0.9894937294337077, 0.5435222109065756, 0.2605499494298821, 0.18895840571753952, 0.2146393825538615, 0.38767723097711493, 0.5397647870456841, 0.1453306559524008, 0.9858618972646433, 0.24099382146241566, 0.35313489768065065, 0.1869337628808425, 0.7115977105340349, 0.5880098094053461, 0.11448284570899614, 0.6200027403661148, 0.79047416050346, 0.058882355981702106, 0.4900062000550598, 0.6406872945214114, 0.5092459530521345, 0.8603096614625256, 0.5518435717546256, 0.08690854928850023, 0.830055502020053, 0.26079192199583956, 0.9445147371808351, 0.11818044658346893, 0.9614416875105056, 0.8154912635400435, 0.5455457075026081, 0.3462818706299028, 0.6870204004211248, 0.2839053847165435, 0.017920951367396687, 0.31034369821216223, 0.25386353284416174, 0.7122960197825169, 0.1574506377425171, 0.7359978743638151, 0.41258782380492576, 0.9089288479619382, 0.4376452383962053, 0.6757517761506807, 0.1267264810161467, 0.9715706325835741, 0.20043763958075456, 0.30710979385881976, 1, 3, 0, 3, 0, 1, 0.04408506921416666, 0.8331627931543505, 0.4093978243977747, 0.413760317015628, 0.15026952092994117, 0.8274351242487543, 0.5558461132798829, 0.8214161934801737, 0.053027477112157384, 0.38556683171997685, 0.7194699847088697, 0.7177087439505272, 0.13336948869193932, 0.6228133763197606, 0.11205242869496668, 0.2933709563320793, 0.7357392487271015, 0.9513167335655217, 0.93430387164439, 0.4464806054218604, 0.4025190780998963, 0.021778213653214973, 0.19416703416803216, 0.8506266978201171, 0.36772506098343016, 0.1733475451954556, 0.8382092293260968, 0.1925925413300995, 0.7833443336193489, 0.3275292604622837, 0.44222692510925543, 0.9728949058304666, 0.6563827652411458, 0.3012997472432788, 0.8959882123268782, 0.16160033100165938, 0.6064385853631366, 0.4563141406539716, 0.31944190711153975, 0.3521536112902246, 0.6901950707428429, 0.0877336177375726, 0.03890288163851996, 0.259340072342909, 0.20173609619709432, 0.07796726442249402, 0.7051402917421268, 0.28096019502761105, 0.6477690449439233, 0.9844504458077292, 0.5110530092767898, 0.6778797142624583, 0.8870103547766228, 0.8404086243247366, 0.9938780436337603, 0.18374735018898702, 0.413648252021094, 0.024317429797932055, 0.28317346388346065, 0.5437611231159, 0.7077161695142352, 0.14960117035524056, 0.5872372997680331, 0.6960668086650446, 0.3237500258814502, 0.28059587666873453, 0.4624604439943186, 0.494306581897235, 0.3001553171200585, 0.11999767984222065, 0.2191698160843828, 0.12494713774051136, 0.6358563893876107, 0.8688484409411642, 0.9017988545634753, 0.9101485924452315, 0.8207684605287788, 0.5404851102184035, 0.03728361649586265, 0.33632946669135677, 0.6114351947531413, 0.8319507325688239, 0.5789481316607455, 0.2560193562941975, 0.6130223852067893, 0.14227160562644803, 2, 2, 3, 2, 2, 1]


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

                if (NOISE_SENSOR == False):
                    # Multiply with the gain
                    agent.inputs[0] = agent.inputs[0] * agent.gainSensor
                    agent.inputs[1] = agent.inputs[1] * agent.gainSensor
                    agent.inputs[2] = agent.inputs[2] * agent.gainSensor
                    agent.inputs[3] = agent.inputs[3] * agent.gainSensor
                else:
                    ruido = agent.inputs[0] * NOISE_DEVIATION
                    agent.inputs[0] = random.uniform(agent.inputs[0] - ruido, agent.inputs[0] + ruido) * agent.gainSensor
                    ruido = agent.inputs[1] * NOISE_DEVIATION
                    agent.inputs[1] = random.uniform(agent.inputs[1] - ruido, agent.inputs[1] + ruido) * agent.gainSensor
                    ruido = agent.inputs[2] * NOISE_DEVIATION
                    agent.inputs[2] = random.uniform(agent.inputs[2] - ruido, agent.inputs[2] + ruido) * agent.gainSensor
                    ruido = agent.inputs[3] * NOISE_DEVIATION
                    agent.inputs[3] = random.uniform(agent.inputs[3] - ruido, agent.inputs[3] + ruido) * agent.gainSensor

                # Make CTRNN RUN
                for i in range(0, N_NEURONAS):
                    change = -agent.outputs[i]

                    for j in range(0, N_NEURONAS):
                        temp = agent.outputs[j] + agent.vBias[j]
                        change += agent.vW[j][i] * sigmoid(temp)

                    change = change + agent.inputs[i]
                    change = change / agent.vTau[i]

                    agent.outputs[i] = agent.outputs[i] + (change * TIME_STEP)

                    activaciones[agent.id][i].append(agent.outputs[i])

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

                            if (NOISE_WEIGHT == True):
                                ruido = weight * NOISE_DEVIATION
                                weight = random.uniform(weight - ruido, weight + ruido)

                            if (NOISE_HOMEOSTATIC == True):
                                ruido = iN * NOISE_DEVIATION
                                iN = random.uniform(iN - ruido, iN + ruido)

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

        # At the end of current light, save the final position and the final distance
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
    # for j in range(0, N_LUCES):
    #     plt.plot(lastPositionX[0][j], lastPositionY[0][j], "bo")
    #     # plt.annotate(j, (lastPositionX[0][j], lastPositionY[0][j]))
    # for j in range(0, N_LUCES):
    #     plt.plot(lastPositionX[1][j], lastPositionY[1][j], "ro")
    #     # plt.annotate(j, (lastPositionX[1][j], lastPositionY[1][j]))
    # for j in range(0, N_LUCES):
    #     plt.plot(lastPositionX[2][j], lastPositionY[2][j], "go")
    #     # plt.annotate(j, (lastPositionX[2][j], lastPositionY[2][j]))
    # for j in range(0, N_LUCES):
    #     plt.plot(lastPositionX[3][j], lastPositionY[3][j], "co")
    #     # plt.annotate(j, (lastPositionX[3][j], lastPositionY[3][j]))
    # for j in range(0, N_LUCES):
    #     plt.plot(lastPositionX[4][j], lastPositionY[4][j], "ko")
    #     # plt.annotate(j, (lastPositionX[4][j], lastPositionY[4][j]))
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
    # # Save activation values of each neuron of each agent
    # for i in range(0, N_AGENTES):
    #     file = open("Results2/activaciones" + str(i) + ".txt", "w")
    #     for j in range(0, N_NEURONAS):
    #         file.write(str(activaciones[i][j]) + "\n")
    #         file.write("---------------------------------------------- \n")
    #     file.close()
    #
    # # Save light positions
    # file = open("Results2/luces.txt", "w")
    # for i in range(0, N_LUCES):
    #     file.write(str(positionLuzX[i]) + " - " + str(positionLuzY[i]) + "\n")
    # file.close()
    #
    # # Save agents paths
    # for i in range(0, N_AGENTES):
    #     file = open("Results2/trayectorias" + str(i) + ".txt", "w")
    #     file.write(str(trayectoriasX[i]) + "\n")
    #     file.write("---------------------------------------------- \n")
    #     file.write(str(trayectoriasY[i]) + "\n")
    #     file.close()
    #
    # # Save agents ending points
    # for i in range(0, N_AGENTES):
    #     file = open("Results2/endingPoints" + str(i) + ".txt", "w")
    #     for j in range(0, N_LUCES):
    #         file.write(str(lastPositionX[i][j]) + " - " + str(lastPositionY[i][j]))
    #     file.close()
