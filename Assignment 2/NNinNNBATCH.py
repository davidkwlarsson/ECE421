import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import time
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    
def buildGraph():
    X = tf.placeholder(tf.float32,[None, 28,28,1], name="X")
    y = tf.placeholder(tf.float32,[None, 10], name="y")
    W1 = tf.get_variable("weights",dtype = tf.float32, shape=[3,3,1,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b1 = tf.get_variable("biases",dtype = tf.float32, shape=[32], initializer=tf.contrib.layers.xavier_initializer())
    W2 = tf.get_variable("W2",dtype = tf.float32, shape=[6272,784], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b2 = tf.get_variable("b2", dtype = tf.float32,shape=[784], initializer=tf.constant_initializer(1))    
    W3 = tf.get_variable("W3", dtype = tf.float32, shape=[784,10], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b3 = tf.get_variable("b3",dtype = tf.float32, shape=[10], initializer=tf.constant_initializer(1))
    
    conv = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.nn.bias_add(conv,b1)
    conv_relu = tf.nn.relu(conv)

    mean, var = tf.nn.moments(conv_relu, axes = [0])
    x1 = tf.nn.batch_normalization(conv_relu, mean,var,1,0,1e-3)
    x1 = tf.nn.max_pool(x1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
    x1_flat = tf.layers.flatten(x1)
    
    s2 = tf.matmul(x1_flat,W2)
    s2 = tf.nn.bias_add(s2,b2)
    x2 = tf.nn.relu(s2)

    s3 = tf.matmul(x2,W3)
    s3 = tf.nn.bias_add(s3,b3)
    
    y_pred = tf.nn.softmax(s3)

    CE = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y,logits = s3), name='cross_entropy')
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001, epsilon = 1e-04)
    train = optimizer.minimize(loss=CE)
    return X,y,W1,W2,W3,b1,b2,b3,y_pred, CE,train

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()   
newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget) 
#tf.reset_default_graph()
epochs = 50
batch_size = 32
number_of_batches = len(newtrain)/batch_size
trainerror = np.zeros(epochs)
trainData = trainData.reshape((-1,28,28,1))


X, y,W1,W2,W3,b1,b2,b3,y_pred, CEloss, train = buildGraph()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run([init,tf.local_variables_initializer()])
for i in range(0,epochs):
    trainData, newtrain = shuffle(trainData,newtrain)
    for batch_nbr in range(0,int(number_of_batches)):
        batch_trainData = trainData[batch_nbr*batch_size+1:(batch_nbr+1)*batch_size]                               
        batch_trainTarget = newtrain[batch_nbr*batch_size+1:(batch_nbr+1)*batch_size]
        _,currentLoss = sess.run([train,CEloss], feed_dict={X: batch_trainData, y: batch_trainTarget})
    if number_of_batches%int(number_of_batches) != 0:
        batch_trainData = trainData[int(number_of_batches)*batch_size+1:]   
        batch_trainTarget = newtrain[int(number_of_batches)*batch_size+1:]
        _,currentLoss = sess.run([train,CEloss], feed_dict={X: batch_trainData, y: batch_trainTarget})
    if i % 10 == 0:
        print("Iteration: %d, error-training: %.5f" %(i, currentLoss))
        
    trainerror[i] = currentLoss
#testData = testData.reshape((-1,28,28,1))
#validData = validData.reshape((-1,28,28,1))
#errTest = sess.run(error, feed_dict = {X: testData, y: newtest})
#errValid = sess.run(error, feed_dict = {X: validData, y: newvalid})
#print("Final training Error: %.5f:" % (trainerror[-1])) 
#print("Final Validation Error: %.5f:" % (errValid))  
#print("Final Testing Error: %.5f:" % (errTest)) 