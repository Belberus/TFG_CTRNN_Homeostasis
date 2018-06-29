import matplotlib.pyplot as plt


array = []

if __name__ == "__main__":
    plt.axhline(y=4.0, color='black', linestyle='-')
    plt.axhline(y=-4.0, color='black', linestyle='-')

    a = []
    i = 0
    while (i < len(array)):
        a.append(array[i])
        i = i + 100

    plt.plot(a, "b-")
    plt.show()
