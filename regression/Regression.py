import numpy as np
import matplotlib.pyplot as plt

# convert data into array


def FileToArray(file):
    filet = open(file, 'r')
    Array = []
    for line in filet:
        num, ex = line.split("e")
        G = float(num) * 10**float(ex)
        Array.append(G)
    return Array


def Convert2d(a1, a2):
    a = np.stack((a1, a2), axis=1)
    return a

    # m is slope | c is y-intersept


def Function(X, m, c):
    X = float(X)
    m = float(m)
    c = float(c)
    Y = m * X + c

    return Y


def slope(p1, p2):
    np.array(p1)
    np.array(p2)
    diff = np.subtract(p2, p1)
    m = diff[1] / diff[0]
    return m


def Yinter(p1, p2):
    m = slope(p1, p2)
    c = p1[1] - m * p1[0]
    return c


def getLine(p1, p2):
    m = slope(p1, p2)
    c = p1[1] - m * p1[0]
    return m, c

# P = [x,y] | L = [m,c]


def SqDistPL(P, L):
    Yp = P[1]
    Yl = Function(P[0], L[0], L[1])
    D = (Yl - Yp)**2
    return D


def Loss(Points, L):
    J = 0
    for P in Points:
        tmp = SqDistPL(P, L)
        J += tmp
    return J


def randLine(Points):
    # if sorted
    p1 = Points[0]
    p2 = Points[-1]
    m, c = getLine(p1, p2)
    return m, c


def changeJm(Points, m, c, step=0.01):
    o = Loss(Points, [m, c])
    a = Loss(Points, [m + step, c])
    change, waste = getLine([m, o], [m + step, a])
    return change


def changeJc(Points, m, c, step=0.01):
    o = Loss(Points, [m, c])
    a = Loss(Points, [m, c + step])
    change, waste = getLine([c, o], [c + step, a])
    return change


def descendM(Points, m, c, step=0.0001):
    if changeJm(Points, m, c) < 0:
        M = m + step
    else:
        M = m - step
    return M


def descendC(Points, m, c, step=0.0001):
    if changeJc(Points, m, c) < 0:
        C = c + step
    else:
        C = c - step
    return C


def findMinM(Points, m, c, acc=4):
    Min = m
    for a in range(acc):
        z = 100**-a
        # for i in range(iterations):
        while True:
            tmp = descendM(Points, Min, c, step=z)
            if Loss(Points, [tmp, c]) < Loss(Points, [Min, c]):
                Min = tmp
            else:
                break
    return Min


def findMinC(Points, m, c, acc=4):
    Min = c
    for a in range(acc):
        z = 100**-a
        # for i in range(iterations):
        while True:
            tmp = descendC(Points, m, Min, step=z)
            if Loss(Points, [m, tmp]) < Loss(Points, [m, Min]):
                Min = tmp
            else:
                break
    return Min


def Optimize(Points, its=3, acc=5, show_loading_bar=False):
    M, C = randLine(Points)
    for i in range(its):
        M = findMinM(Points, M, C, acc)
        C = findMinC(Points, M, C, acc)
    return M, C


"""# find best slope and y-intersept
M, C = Optimize(data, show_loading_bar=False)

# create points to plot
t = [1, 10]
v = [Function(1, M, C), Function(10, M, C)]

plt.scatter(data[:, 0], data[:, 1])  # plot data
plt.plot(t, v)  # plot line

print("Slope:", M, "\nC:", C)
plt.xlabel('Age')
plt.ylabel('Height[M]')
plt.title('Correlation between height and age')
"""

# bonus


def ToyData(m, c, spread=1, length=10, dist=1, size=70):
    S = [[1.36437996,  1.93244223],
         [1.77090512,  2.89132355]]
    for l in range(length):
        s = (spread * np.random.randn(70, 2))
        s[:, 0] += l * dist
        s[:, 1] += Function(l * dist, m, c)

        S = np.concatenate((S, s))

    return S


m = 5
c = 1

S = ToyData(m, c, spread=0.2, length=10, dist=0.2)

M, C = Optimize(S, its=10, acc=8, show_loading_bar=True)

# boundaries for line
lowx = min(S[:, 0])
hix = max(S[:, 0])


t = [lowx, hix]
v = [Function(lowx, M, C), Function(hix, M, C)]
plt.scatter(S[:, 0], S[:, 1])
plt.plot(t, v, "r")
print("Slope:", M, "\nC:", C)
print("\nreal\nSlope:", m, "\nC:", c)
plt.show()

"""X = FileToArray("ex2x.dat")
Y = FileToArray("ex2y.dat")
data = Convert2d(X, Y)
plt.scatter(data[:, 0], data[:, 1])
M, C = Optimize(data)

lowx = min(data[:, 0])
hix = max(data[:, 0])
t = [lowx, hix]
v = [Function(lowx, M, C), Function(hix, M, C)]
plt.plot(t, v, "r")
print("Slope:", M, "\nC:", C)

plt.xlabel('Age')
plt.ylabel('Height[M]')
plt.title('Correlation between height and age')
"""
plt.show()
