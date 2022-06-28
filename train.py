import math
import random
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig


# creat data set
X = np.arange(-1*math.pi, 1*math.pi, 0.006)  # 0.006
Y = np.sin(X)
datasize = np.size(X)
# creat hide parameter array
hidesize = 10
W1 = np.random.random((hidesize, 1))
B1 = np.random.random((hidesize, 1))
W2 = np.random.random((1, hidesize))
b2 = np.random.random(1)
yita = 1
loop = 5000  # 5000
Y_predict = np.zeros(datasize)
LOOP = []
E = []
for loopi in range(loop):
    tempsume = 0
    for i in range(datasize):
        x = X[i]
        Z1 = W1*x + B1
        A1 = sigmoid(Z1)
        z2 = W2*A1 + b2
        Y_predict[i] = z2[0, 0]
        e = Y[i] - Y_predict[i]
        dW2 = -1*A1.T
        db2 = -1
        dW1 = sigmoid(Z1)*(np.full_like(x, 1)-sigmoid(Z1))*x
        dB1 = sigmoid(Z1)*(np.full_like(x, 1)-sigmoid(Z1))
        W2 = W2-dW2*yita*e
        b2 = b2-db2*yita*e
        W1 = W1-dW1*yita*e
        B1 = B1-dB1*yita*e

        tempsume = tempsume + abs(e)

    # if loopi % 10 is 0:
    #     print("loop:", loopi, "\ttempsume:", tempsume)
    LOOP.append(loopi)
    E.append(tempsume)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    plt.sca(ax1)
    plt.plot(LOOP, E, color='black')
    plt.sca(ax2)
    plt.plot(X, Y)
    plt.plot(X, Y_predict, linewidth=1, linestyle='--')
    plt.pause(0.001)  # 暂停0.01秒
    plt.cla()
    plt.ioff()  # 关闭画图的窗口
    # print("W1", W1)
    # print("B1", B1)
    # print("W2", W2)
    # print("b2", b2)
