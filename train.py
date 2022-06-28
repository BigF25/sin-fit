import math
import random
from turtle import end_fill
import numpy as np

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
    else:
        z = math.exp(x)
        sig = z / (1 + z)
    return sig

# creat data set
X = np.arange(0, 2*math.pi, 0.006)#0.006
Y = np.sin(X)
datasize = np.size(X)
# creat hide parameter array
hidesize = 5
W1 = np.random.random((hidesize,1))
B1 = np.random.random((hidesize,1))
W2 = np.random.random((1,hidesize))
b2 = np.random.random(1)
yita = 0.005
loop = 5000 #5000
E = np.zeros([1,loop])
Y_predict = np.zeros(datasize)
for loopi in range(loop):
    tempsume = 0
    for i in range(datasize):
        x = X[i]
        Z1 = W1*x +B1
        A1 = np.zeros([hidesize,1])
        for j in range(hidesize):
            A1[j,0] = sigmoid(Z1[j,0])
        z2 = W2@A1 + b2
        # a2 = sigmoid(z2[0,0])
        Y_predict[i] = z2[0,0]

        e = Y[i] - Y_predict[i]
        # dW2 = e*yita*A1
        dW2 = np.zeros([1,hidesize])
        for j in range(hidesize):
            dW2[0,j] = -1*A1[j,0]
        # db2 = -1*yita*e
        db2 = -1
        dW1 = np.zeros([hidesize,1])
        for j in range(hidesize):
            # dW1[j,0] = W2[0,j]*sigmoid(Z1[j,0])*(1-sigmoid(Z1[j,0]))*x*e*yita
            dW1[j,0] = sigmoid(Z1[j,0])*(1-sigmoid(Z1[j,0]))*x
        dB1 = np.zeros([hidesize,1])
        for j in range(hidesize):
            # dB1[j,0] = W2[0,j]*sigmoid(Z1[j,0])*(1-sigmoid(Z1[j,0]))*(-1)*e*yita
            dB1[j,0] = sigmoid(Z1[j,0])*(1-sigmoid(Z1[j,0]))
        W2 = W2-dW2*yita*e
        b2 = b2-db2*yita*e
        W1 = W1-dW1*yita*e
        B1 = B1-dB1*yita*e
    
        tempsume = tempsume + abs(e)

        # print("W2",W2)
        # print("b2",b2)
        # print("W1",W1)
        # print("B1",B1)
    E[0,loopi] = tempsume
    if loopi%5 is 0:
        print("loop:",loopi,"\ttempsume:",tempsume)
print("W1",W1)
print("B1",B1)        
print("W2",W2)
print("b2",b2)

