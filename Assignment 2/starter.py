import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def relu(x):
    # TODOx[x < 0] = 0
    np.maximum(x,0)
    return x

def softmax(x):
    # TODO
    
    x = x - np.amax(x)
    sigma = np.exp(x)/(np.sum(np.exp(x),axis = 1,keepdims = True))
    return sigma
    
def computeLayer(x, w, b):
    # TODO
    return np.matmul(x,w) + b

def avrageCE(target, prediction):
    # TODO
    s = softmax(prediction)
    #axis = 1 sums along the columns --> over k
    return -np.mean(np.sum(target*np.log(s), axis = 1)) 
    

def gradCE(target, prediction):
    s = softmax(prediction)

    # TODO
    return -np.mean(np.divide(target,s),axis = 0,keepdims = True)

def ReLU_prim(x):
    x[x<0] = 0
    x[x>=0] = 1
    return x
    
def softmax_prim(s2):
#    prim = softmax(x)*(np.ones(x.shape)-softmax(x))
#    s2 = np.mean(s2,axis=0,keepdims=True)
    sm2 = softmax(s2)
    grad = -sm2*sm2
    for i in range(s2.shape[1]):
        grad[i][i] = sm2[0][i]*(1-sm2[0][i])
#        
    return grad
#    return prim
    
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()    
newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget) 
trainData = trainData.reshape(-1,28*28)

epochs = 5
hiddenunits = 1000
classes = 10
gamma = 0.9
alpha = 0.005
d0 = len(trainData[1,:])
v2 = 1e-5 * np.ones((hiddenunits,classes))
v1 = 1e-5 * np.ones((d0,hiddenunits))
w2 = np.random.normal(0,np.sqrt(2)/np.sqrt(hiddenunits+classes),(hiddenunits,classes))
w1 = np.random.normal(0,np.sqrt(2)/np.sqrt(hiddenunits+d0),(d0,hiddenunits))
b1 = np.sqrt(2)*np.ones(hiddenunits)/np.sqrt(hiddenunits+d0)
b2 = np.sqrt(2)*np.ones(classes)/np.sqrt(hiddenunits+classes)

for i in range(epochs):
    print(w2)
    print(w1)
    s1 = computeLayer(trainData, w1, b1)
    x1 = relu(s1)
    s2 = computeLayer(x1,w2,b2)
    x2 = softmax(s2)
    delta2 = gradCE(newtrain,s2)*softmax_prim(s2)
    print(softmax_prim(s2).shape)
    v2 = gamma*v2 + alpha*np.matmul(x1.T,delta2)
    w2 = w2 - v2
    b2 = b2 - alpha*np.matmul(np.ones(10000),delta2)

    delta1 = np.matmul(delta2,w2.T)*ReLU_prim(s1)

    v1 = gamma*v1 + alpha*np.matmul(trainData.T,delta1)
    w1 = w1 - v1
    b1 = b1 - alpha*np.matmul(np.ones(10000),delta1)
    