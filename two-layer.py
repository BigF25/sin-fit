import math
import random
import numpy as np
import matplotlib.pyplot as plt


'''define sigmoid function'''
def sigmoid(x):
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig


'''creat data set'''
X = np.arange(0, 2*math.pi, 0.006)#0.006
Y = np.sin(X)
datasize = np.size(X)
'''creat hide parameter'''
hidesize = 10
W1 = np.random.random((hidesize,1))
B1 = np.random.random((hidesize,1))
W2 = np.random.random((1,hidesize))
B2 = np.random.random(1)
'''creat other parameter'''
yita = 0.005
loop = 5000 #5000
E = []
LOOP = []
Y_hat = np.zeros(datasize)
'''start train'''
for loopi in range(loop):
    tempsume = 0
    for i in range(datasize):
        '''forward propagation'''
        x = X[i]
        Z1 = W1*x +B1
        A1 = sigmoid(Z1)
        Z2 = W2@A1 + B2
        Y_hat[i] = Z2[0,0]
        e = Y[i] - Y_hat[i]
        '''backword propagation'''
        dW2 = -1*A1.T
        dB2 = -1
        dW1 = sigmoid(Z1)*(1-sigmoid(Z1))*x
        dB1 = sigmoid(Z1)*(1-sigmoid(Z1))
        W2 = W2-dW2*yita*e
        B2 = B2-dB2*yita*e
        W1 = W1-dW1*yita*e
        B1 = B1-dB1*yita*e

        tempsume = tempsume + abs(e)

    '''print cost'''
    if loopi%5 is 0:
        print("loop:",loopi,"\ttempsume:",tempsume)
    '''validate the model'''    
    X2 = np.arange(0, 2*math.pi, 0.06) 
    Y2 = np.sin(X2)
    size2 = np.size(X2)
    Y_predict = np.zeros(size2)
    for j in range(size2):
        x = X2[j]
        Z1 = W1*x +B1
        A1 = sigmoid(Z1)
        Z2 = W2@A1 + B2
        Y_predict[j] = Z2[0,0]
    LOOP.append(loopi)
    E.append(tempsume)
    ax1 = plt.subplot(1, 2, 1)
    plt.sca(ax1)
    plt.plot(LOOP, E, color='black')
    ax2 = plt.subplot(1, 2, 2)
    plt.sca(ax2)
    plt.plot(X2, Y2, '.')
    plt.plot(X2, Y_predict, '.')
    plt.pause(0.01)  # 暂停0.01秒
    plt.cla()
    plt.ioff()  # 关闭画图的窗口

'''print the best model'''
print("W1",W1)
print("B1",B1)        
print("W2",W2)
print("b2",B2)