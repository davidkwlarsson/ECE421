# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 11:37:17 2019

@author: David_000
"""

import numpy as np
import numpy.linalg as npl
import tensorflow as tf

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
    return npl.sum((npl.norm(np.inner(W,x) + b - y)^2)/(2*N) + reg/2*np.inner(W,W))
    
def grad_MSE(W,b,x,y,reg):
    
    return 0
    
def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol): 
    return 0
    
def crossEntropyLoss(W, b, x, y, reg):
    return 0
    
def gradCE(W, b, x, y, reg):
    return 0
