import matplotlib.pyplot as plt

lucesX = [-58.725238932010235, -85.9224578630104, -76.87532599355573, -75.64836608238488, -46.81903926746752, -104.52358407452692]
lucesY = [74.85163106400668, 87.94173023632855, 28.652961297936194, -95.10497864626953, -46.06440382859642, -91.5240516454823]

tX = []

tY = [+]


if __name__ == "__main__":
    plt.scatter(lucesX, lucesY, s=60, c='red', marker='^')
    for i in range(0, len(lucesX)):
        plt.annotate(i, (lucesX[i],lucesY[i]))

    plt.plot(tX, tY, "b-")
    plt.scatter(tX[0], tY[0], s=80, c='black', marker='o')
    plt.show()
