# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:48:14 2019

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
    N,n,m = trainData.shape
    trainData = trainData.reshape(N,n*m)
    testData = testData.reshape(testData.shape[0],testData.shape[1]*testData.shape[2])
    validData = validData.reshape(validData.shape[0],validData.shape[1]*validData.shape[2])

def buildGraph(loss=None): 
    
    #Initialize weight and bias tensors 
    X = tf.placeholder(tf.float32,[None, 784], name="X")
    y = tf.placeholder(tf.float32,[None, 1], name="y")
    reg = tf.placeholder(tf.float32, name="reg")
    
    W = tf.Variable(tf.truncated_normal(shape=[784,1], stddev=0.5), name='weights')
    b = tf.Variable(0.0, name='biases')
    tf.set_random_seed(421)
    if loss == "MSE":
        y_pred = tf.matmul(X,W) + b
        meanSquaredError = tf.reduce_mean(tf.reduce_mean(tf.square(y_pred - y), 
                                                reduction_indices=1, 
                                                name='squared_error'), 
                                  name='mean_squared_error') + reg/2*tf.norm(W)
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, epsilon = 1e-04)
        train = optimizer.minimize(loss=meanSquaredError)
        return W, b, X, y, reg, y_pred, meanSquaredError, train
      
    elif loss == "CE": 
        y_pred = 1/(1+tf.exp(-(tf.matmul(X,W) + b)))
        crossEntropy = tf.reduce_mean(tf.reduce_mean((1-y)*(tf.matmul(X,W) + b) + tf.log(1+tf.exp(-(tf.matmul(X,W) + b))), 
                                                reduction_indices=1, 
                                                name='probabilty_error'))
        optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, epsilon = 1e-04)
        train = optimizer.minimize(loss=crossEntropy)
        return W, b, X, y, reg, y_pred, crossEntropy, train
        
    return 0
    
    
epochs = 700
batch_size = 500
number_of_batches = N/batch_size
reg0 = 0
trainerror = np.zeros(epochs)

#W, b, X, y, y_pred, crossEntrophy, train = buildGraph("CE")
#init = tf.global_variables_initializer()
#sess = tf.InteractiveSession()
#sess.run(init)
#for step in range(0, 200):
#    _, err, currentW, currentb, yhat = sess.run([train, crossEntrophy, W, b, y_pred], feed_dict={X: trainData, y: trainTarget}) 
#    if step % 10 == 0:
#        print("Iteration: %d, MSE-training: %.2f" %(step, err))
            
W, b, X, y, reg, y_pred, error, train = buildGraph("CE")
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
for step in range(0, epochs):
    #shuffle the trainData and trainTarget
    shuffle = np.arange(0,N-1)
    np.random.shuffle(shuffle)
    trainData = trainData[shuffle]
    trainTarget = trainTarget[shuffle]
    #divide it into minibatches and do the optimization for each minibatch
    for batch_nbr in range(0,int(number_of_batches)):
        batch_trainData = trainData[batch_nbr*batch_size+1:(batch_nbr+1)*batch_size]
        batch_trainTarget = trainTarget[batch_nbr*batch_size+1:(batch_nbr+1)*batch_size]
        _, err, currentW, currentb, yhat = sess.run([train, error, W, b, y_pred], feed_dict={X: batch_trainData, y: batch_trainTarget, reg: reg0})
    if number_of_batches%int(number_of_batches) != 0:
        batch_trainData = trainData[batch_nbr*batch_size+1:]
        batch_trainTarget = trainTarget[batch_nbr*batch_size+1:]
        _, err, currentW, currentb, yhat = sess.run([train, error, W, b, y_pred], feed_dict={X: batch_trainData, y: batch_trainTarget, reg: reg0})
    if step % 10 == 0:
        print("Iteration: %d, error-training: %.5f" %(step, err))
#        trainerror = np.append(trainerror, err)
    trainerror[step] = err
errTest = sess.run(error, feed_dict = {X: testData, y: testTarget, reg : reg0})
errValid = sess.run(error, feed_dict = {X: validData, y: validTarget, reg : reg0})
print("Final training Error: %.5f:" % (trainerror[-1])) 
print("Final Validation Error: %.5f:" % (errValid))  
print("Final Testing Error: %.5f:" % (errTest))    

fig, ax = plt.subplots()
ax.plot(trainerror)

ax.set(xlabel='iterations', ylabel='trainerror',
       title='Error from the training Data')
ax.grid()

plt.show()    