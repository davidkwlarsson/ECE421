# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:47:08 2019

@author: David_000
"""
import numpy as np
import numpy.linalg as npl
import tensorflow as tf
    
    
def crossEntropyLoss(W, b, x, y, reg):
    CEloss_d = np.mean((1-y)*(np.matmul(x,W) + b) + np.log(1+np.exp(-(np.matmul(x,W) + b))))
    CEloss = CEloss_d + reg/2*npl.norm(W)**2
    return CEloss
    
def gradCE(W, b, x, y, reg):
    nablaw = np.mean((x - x/(1+np.exp(np.matmul(x,W) + b)) - y*x))
    nablab = np.mean((1-y) + (-1)/(1+np.exp(np.matmul(x,W) + b)))
    
    return nablaw,nablab