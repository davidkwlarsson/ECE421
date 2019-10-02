# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:37:17 2019

@author: David_000
"""

import numpy as np
import numpy.linalg as npl
import tensorflow as tf
import matplotlib.pyplot as plt

with np.load('notMNIST.npz') as data : 
    Data, Target = data ['images'], data['labels']
    posClass = 2
    negClass = 9 
    dataIndx = (Target==posClass) + (Target==negClass)
    Data = Data[dataIndx]/255.
    Target = Target[dataIndx].reshape(-1, 1)
    Target[Target==posClass] = 1
    Target[Target==negClass] = 0
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data, Target = Data[randIndx], Target[randIndx]
    trainData, trainTarget = Data[:3500], Target[:3500]
    validData, validTarget = Data[3500:3600], Target[3500:3600] 
    testData, testTarget = Data[3600:], Target[3600:]


def MSE(W,b,x,y,reg):
    N = np.size(y,0)
    mse = 1/(2*N)*npl.norm(np.matmul(x,W) + b - y)**2+reg/2*npl.norm(W)**2
    return mse

def grad_MSE(W,b,x,y,reg):
    N = np.size(y,0)
    nablaw = np.matmul(np.transpose(x),(np.matmul(x,W) + b - y))/N + W*reg
    nablab = np.mean(np.matmul(x,W) + b - y)
    return nablaw, nablab
    
def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, lossType="None"):
    loss = np.zeros(epochs)
    if lossType == "CE":
        for i in range(epochs):
            gt = grad_CE(W,b,x,y,reg)
            print(alpha*npl.norm(gt[0]))
            Wopt = W - alpha * np.reshape(gt[0],(len(W),-1))
            b = b - alpha * gt[1]
            
            W = Wopt
            loss[i] = crossEntropyLoss(W,b,x,y,reg)
            if npl.norm(alpha*gt[0]) < error_tol:
                return Wopt,b,loss[:i]

    elif lossType == "MSE":        
        
        for i in range(epochs): 
    #        print(i)
            gt = grad_MSE(W,b,x,y,reg)
            Wopt = W - alpha * gt[0]
            b = b - alpha * gt[1]
            W = Wopt
            loss[i] = MSE(W,b,x,y,reg)
            if npl.norm(alpha*gt[0]) < error_tol:
                return Wopt,b,loss[:i]
            
    return Wopt,b,loss

    
def crossEntropyLoss(W, b, x, y, reg):
    CEloss_d = np.mean((1-y)*(np.matmul(x,W) + b) + np.log(1+np.exp(-(np.matmul(x,W) + b))))
    CEloss = CEloss_d + reg/2*npl.norm(W)**2
    return CEloss
    
def grad_CE(W, b, x, y, reg):
    nablaw = np.matmul(x.T,(1-y) - 1/(1+np.exp(np.matmul(x,W) + b)))/len(y) + reg*W
    nablab = np.mean((1-y) + (-1)/(1+np.exp(np.matmul(x,W) + b)))
    return nablaw,nablab
    
    
def calcAccuracy(W, b, x, y, regType="Log"):
    matrixMult = (np.matmul(x,W) + b)
    if regType == "Lin":
        y2 = matrixMult
        correct = 0
        for i in range(len(y)):
            guess = 0
            if y2[i] > 0.5:
                guess = 1
            if guess == y[i]:
                correct += 1
        
        return correct/len(y)
    else:
        y2 = 1/(1+np.exp(-matrixMult))
        correct = 0
        for i in range(len(y)):
            guess = 0
            if y2[i] > 0.5:
                guess = 1
            if guess == y[i]:
                correct += 1
        return correct/len(y)
            
            




epochs = 5000
reg = 0.1
alpha = 0.005
error_tol = 1e-07

N,n,m = trainData.shape
W0 = np.zeros((n*m,1)) #inital guess att zeros
b = 0;
trainData = trainData.reshape(N,n*m)
Wopt = grad_descent(W0,b,trainData,trainTarget,alpha,epochs,reg,error_tol,"MSE")
W = Wopt[0]
b = Wopt[1]
trainError = Wopt[2]
#plt.plot(Wopt[2])

#Now Calculate Validation and test error.
validData = validData.reshape(len(validTarget), np.size(validData,1)*np.size(validData,2))
testData = testData.reshape(len(testTarget), np.size(testData,1)*np.size(testData,2))

print(MSE(W,b,trainData,trainTarget,reg))
print(MSE(W,b,validData,validTarget,reg))
print(MSE(W,b,testData,testTarget,reg))

fig, ax = plt.subplots()
ax.plot(trainError)

ax.set(xlabel='iterations', ylabel='trainerror',
       title='Error from the training Data')
ax.grid()

plt.show()    
#validError = MSE(W,b,validData,validTarget,reg)
#testError = MSE(W,b,testData,testTarget,reg)