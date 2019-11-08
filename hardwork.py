import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def readData(filename):
    df = pd.read_csv(filename)
    return df.values

x = readData("./Linear_X_Train.csv")
y = readData("./Linear_Y_Train.csv")

#print(x.shape)
#print(y.shape)
#plt.scatter(x,y)
x = x.reshape(3750,)
y = y.reshape(3750,)
X = (x-x.mean())/(x.std())
Y = y
#plt.scatter(X,Y)
#plt.show()


def hypothesis(theta,x):
    return theta[0]+theta[1]*x

def error(X,Y,theta):
    total_error = 0
    m = X.shape[0]
    for i in range(m):
        total_error += (hypothesis(theta,X[i])-Y[i])**2
    return 0.5*total_error

def gradient(X,Y,theta):
    grad=np.zeros((2,))
    m=X.shape[0]
    for i in range(m):
        grad[0]+=(hypothesis(theta,X[i])-Y[i])
        grad[1]+=(hypothesis(theta,X[i])-Y[i])*X[i]
    return grad

def gradientDescent(X,Y,learning_rate,maxItr):
    grad=np.zeros((2,))
    theta=np.zeros((2,))
    error_list = []
    for i in range(maxItr):
        grad=gradient(X,Y,theta)
        e = error(X,Y,theta)
        error_list.append(e)
        theta[0] = theta[0] - learning_rate*grad[0]
        theta[1] = theta[1] - learning_rate*grad[1]
    return theta,error_list

final_theta,error_list = gradientDescent(X,Y,learning_rate=0.001,maxItr=1000)
#print(theta[0],theta[1])
plt.scatter(X,Y)
plt.plot(X,hypothesis(final_theta,X),color="g")
#plt.plot(error_list)
plt.show()




