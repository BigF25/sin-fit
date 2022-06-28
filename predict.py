#tempsume = 33.61280151433589
import numpy as np
import math
import matplotlib.pyplot as plt
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        sig = 1 / (1 + z)
    else:
        z = math.exp(x)
        sig = z / (1 + z)
    return sig

hidesize = 5
W1 = np.array([[-0.462176],
               [6.3740212],
               [1.51078193],
               [6.21981207],
               [6.14097489]], np.float64)
B1 = np.array([[2.34340023],
               [4.40988409],
               [-5.10746775],
               [4.42198029],
               [4.4281429]], np.float64)
W2 = np.array([[-8.98294215, 1.85818847, - 5.51120176, 2.3838906, 3.2498191]])
b2 = np.array([1.2098803])
X = np.arange(0, 2*math.pi, 0.005)
size = np.size(X)

Y = np.sin(X)
Y_predict = np.zeros(size)

for i in range(size):
    x = X[i]
    Z1 = W1*x +B1
    A1 = np.zeros([hidesize,1])
    for j in range(hidesize):
        A1[j,0] = sigmoid(Z1[j,0])
    z2 = W2@A1 + b2
    Y_predict[i] = z2[0,0]

plt.plot(X,Y)
plt.plot(X,Y_predict,color='red',linewidth=1,linestyle='--')
plt.show()
print(Y_predict)
print(Y)
