import math
import random
import numpy as np
import matplotlib.pyplot as plt


'''define sigmoid function'''
def sigmoid(x):
    sig = np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))
    return sig


'''creat data set'''
X = np.arange(-1*math.pi, 1*math.pi, 0.006)  # 0.006
Y = np.sin(X)
datasize = np.size(X)
'''creat hide parameter array'''
hidesize1 = 10
hidesize2 = 5
W1 = np.random.random((hidesize1, 1))
B1 = np.random.random((hidesize1, 1))
W2 = np.random.random((hidesize2, hidesize1))
B2 = np.random.random((hidesize2, 1))
W3 = np.random.random((1, hidesize2))
B3 = np.random.random((1, 1))
W1_best = np.copy(W1)
B1_best = np.copy(B1)
W2_best = np.copy(W2)
B2_best = np.copy(B2)
W3_best = np.copy(W3)
B3_best = np.copy(B3)
'''creat other parameter'''
tempsume_min = 100
yita = 0.01
loop = 5000 # 5000
Y_hat = np.zeros(datasize)
LOOP = []
E = []
'''start train'''
for loopi in range(loop):
    tempsume = 0
    for i in range(datasize):
        '''forward propagation'''
        x = X[i]
        Z1 = W1*x + B1
        A1 = sigmoid(Z1)
        Z2 = np.dot(W2, A1) + B2
        A2 = sigmoid(Z2)
        Z3 = np.dot(W3, A2) + B3
        e = Y[i] - Z3[0, 0]
        '''backword propagation'''
        dB3 = -1
        dW3 = -1*A2.T
        dB2 = sigmoid(Z2)*(1-sigmoid(Z2))
        dW2 = sigmoid(Z2)*(1-sigmoid(Z2))*A1.T
        dB1 = sigmoid(Z1)*(1-sigmoid(Z1))
        dW1 = sigmoid(Z1)*(1-sigmoid(Z1))*x
        B3 = B3-dB3*yita*e
        W3 = W3-dW3*yita*e
        B2 = B2-dB2*yita*e
        W2 = W2-dW2*yita*e
        B1 = B1-dB1*yita*e
        W1 = W1-dW1*yita*e
        '''conut cost'''
        tempsume = tempsume + abs(e)

    '''save the best model'''
    if tempsume < tempsume_min:
        tempsume_min = tempsume
        W1_best = W1
        B1_best = B1
        W2_best = W2
        B2_best = B2
        W3_best = W3
        B3_best = B3
    '''print cost'''
    # if loopi % 10 is 0:
    #     print("loop:", loopi, "\ttempsume:", tempsume)
    '''validate the model'''
    X2 = np.arange(-1*math.pi, 1*math.pi, 0.1) 
    Y2 = np.sin(X2)
    size2 = np.size(X2)
    Y_predict = np.zeros(size2)
    for j in range(size2):
        x2 = X2[j]
        Z1 = W1*x2 + B1
        A1 = sigmoid(Z1)
        Z2 = np.dot(W2,A1) + B2
        A2 = sigmoid(Z2)
        Z3 = np.dot(W3,A2) + B3
        Y_predict[j] = Z3[0, 0]
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
# print("W1_best", W1_best)
# print("B1_best", B1_best)
# print("W2_best", W2_best)
# print("B2_best", B2_best)
# print("W3_best", W3_best)
# print("B3_best", B3_best)
'''predict'''
# X2 = np.arange(-1*math.pi, 1*math.pi, 0.006) 
# # X = X*2*math.pi-math.pi
# Y2 = np.sin(X2)
# size2 = np.size(X2)
# Y_predict = np.zeros(size2)
# for i in range(size2):
#     x = X2[i]
#     Z1 = W1_best*x + B1_best
#     A1 = sigmoid(Z1)
#     Z2 = np.dot(W2_best,A1) + B2_best
#     A2 = sigmoid(Z2)
#     Z3 = np.dot(W3_best,A2) + B3_best
#     Y_predict[i] = Z3[0, 0]
# plt.plot(X2, Y2, '.')
# plt.plot(X2, Y_predict, '.')
# plt.show()