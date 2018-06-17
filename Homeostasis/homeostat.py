# Alberto Martinez Menendez
#   HOMEOSTAT
#-----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import random
import copy

# N unidades conectadas. Cada unidad recive N inputs de ella y de las otras unidades
# cuyo peso esta definido por la fuerza de la conexion entre ellas. El peso del
# sumatorio I de las inputs sobre una unidad determina su salida s.
# Una unidad estara especidifica como un set de parametros U={w1,...wN,x1,...xN,y1,...yN}
# donde wi es la fuerza de la conexion desde la unidad i-esima y (xj,yj) es la coordenada del
# j-esimo punto en su funcion de transferencia.
# Los rangos son w E [0.00, 1.00] y s E [0.00, 1.00] para que I E [0.00, N]

# Al comienzo los pesos de la fuerza de las conexiones estan distribuidos de forma aleatoria
# dentro de los rangos apropiados R = [0.5 - alpha, alpha + 0.5] donde alpha determina la
# opresion de la restriccion homeostatica. Si s pertenece a R la unidad es homeostatica. Si s
# no pertenece a R entonces la unidad ha perdido homeostasis y se activa el cambio adaptativo.

# Variables globales
N = 4
P = 4
minRange = 0.4    # 0.5 - 0.1 (alpha)
maxRange = 0.6    # 0.5 + 0.1 (alpha)

# Clase unidad
class Unit:
    def __init__(self):
        self.salida =  round(random.uniform(0.00, 1.00),2)
        self.w = [] # Pesos de las conexiones con el resto de unidades
        self.x = [] # Valores X de la funcion de transferencia de esta unidad
        self.y = [] # Valores Y de la funcion de transferencia de esta unidad
        self.s = [] # Salidas de las unidades
        self.historicoS = []    # Historico de salidas de esta unidad para su posterior pintado

    def randomizar(self):
        wUnit = []
        xUnit = []
        yUnit = []

        # Inicializamos los arrays
        for i in range(0, N):
            wUnit.append(0.00)

        for i in range(0, P):
            xUnit.append(0.00)
            yUnit.append(0.00)

        # Randomizamos los arrays
        for i in range(0, N):
            wUnit[i] = round(random.uniform(0.00, 1.00),2)

        for i in range(0, P):
            yUnit[i] = round(random.uniform(0.00, 1.00),2)

        xUnit[0] = 0.00
        xUnit[P-1] = N
        xUnit[1] = round(random.uniform(0.00, N),2)
        xUnit[2] = round(random.uniform(0.00, N),2)

        self.w = wUnit
        self.x = xUnit
        self.y = yUnit

    def calcularS(self):
        # Calculamos el I
        I = 0.0
        for i in range(0, N):
            I += self.w[i] * self.s[i]

        # Calculamos la salida
        if (I >= self.x[0] and I < self.x[1]):
            self.salida = self.y[0] + (self.y[1] - self.y[0]) * ((I - self.x[0]) / (self.x[1] - self.x[0]))
            self.historicoS.append(copy.copy(self.salida))
        elif (I >= self.x[1] and I < self.x[2]):
            self.salida = self.y[1] + (self.y[2] - self.y[1]) * ((I - self.x[1]) / (self.x[2] - self.x[1]))
            self.historicoS.append(copy.copy(self.salida))
        elif (I >= self.x[3] and I < self.x[4]):
            self.salida = self.y[2] + (self.y[3] - self.y[2]) * ((I - self.x[2]) / (self.x[3] - self.x[2]))
            self.historicoS.append(copy.copy(self.salida))


if __name__ == "__main__":
  t = 0

  # Creamos las unidades y las inicialiamos
  units = []

  for i in range(0, N):
    units.append(Unit())

  for i in range(0, N):
    units[i].randomizar()

  # Almacenamos las salidas de las unidades en el resto de unidades
  for i in range(0, N):
      for j in range(0, N):
          units[i].s.append(units[j].salida)

  # Bucle del sistema
  while (t<99):
    for i in range(0, N):
        # Calculamos la nueva salida y cambiamos su valor en las demas unidades
        units[i].calcularS()

        for j in range(0, N):
            units[j].s[i] = units[i].salida

        # Si perdemos la homeostasis ponemos valores aleatorios en los pesos y en los valores X e Y
        if (units[i].salida <= minRange or units[i].salida >= maxRange):
            print ("Desequilibrio")
            units[i].randomizar()

    t+= 1

  print ("Hecho")

  # Pintamos las s de una unidad
  plt.plot(units[0].historicoS)
  plt.plot(units[1].historicoS)
  plt.plot(units[2].historicoS)
  plt.plot(units[3].historicoS)
  plt.ylabel('Outputs')
  plt.axhline(y=minRange, color='black', linestyle='-')
  plt.axhline(y=maxRange, color='black', linestyle='-')
  plt.ylim(0.0,0.8)
  plt.show()
