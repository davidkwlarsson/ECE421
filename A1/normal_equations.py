# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 14:25:47 2019

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
    
    
N,n,m = trainData.shape
trainData = trainData.reshape(N,n*m)
trainData_1 = np.append(np.ones((N,1)),trainData, axis = 1)
validData = validData.reshape(len(validTarget), np.size(validData,1)*np.size(validData,2))
testData = testData.reshape(len(testTarget), np.size(testData,1)*np.size(testData,2))
W_LS = np.matmul(npl.inv(np.matmul(trainData_1.T,trainData_1)),np.matmul(trainData_1.T,trainTarget))
b = W_LS[0]
W = W_LS[1:]
reg = 0
print(MSE(W,b,trainData,trainTarget,reg))
print(MSE(W,b,validData,validTarget,reg))
print(MSE(W,b,testData,testTarget,reg))