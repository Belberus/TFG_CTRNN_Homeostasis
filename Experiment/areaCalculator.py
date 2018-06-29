import math
import matplotlib.pyplot as plt
import numpy as np

# Calcula el area que se ha salido de la zona homeostatica dado un array con las activaciones de la neurona

array = []

MINIMO = -4.0

MAXIMO = 4.0


if __name__ == "__main__":
    a = []
    i = 0
    while (i < len(array)):
        a.append(array[i])
        i = i + 100

    aEncima = a[:]
    aDebajo = a[:]
    for i in range(0, len(a)):
        if aEncima[i] < MAXIMO:
            aEncima[i] = MAXIMO
        aEncima[i] = aEncima[i] - MAXIMO

    for i in range(0, len(a)):
        if aDebajo[i] > MINIMO:
            aDebajo[i] = MINIMO
        aDebajo[i] = aDebajo[i] - MINIMO


    # Integramos para saber las areas de perdida de estabilidad
    porEncima = np.trapz(aEncima, axis = 0)
    porDebajo = np.trapz(aDebajo, axis = 0)
    print(porEncima, " ", porDebajo)
    print(porEncima + (-1 * porDebajo))
