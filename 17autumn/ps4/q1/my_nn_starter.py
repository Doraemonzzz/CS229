# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 17:52:45 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt

def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def softmax(x):
    """
    Compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    """
	### YOUR CODE HERE
    s1 = np.exp(x)
    s = s1 / np.sum(s1, axis=1).reshape(-1, 1)
	### END YOUR CODE
    return s

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    s = 1 / (1 + np.exp(-x))
    ### END YOUR CODE
    return s

def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    #隐藏层
    s1 = data.dot(W1) + b1
    h = sigmoid(s1)
    #输出层
    s2 = h.dot(W2) + b2
    y = softmax(s2)
    cost = - np.mean(labels * np.log(y))
    ### END YOUR CODE
    return h, y, cost

def loss(data, labels, params, Lambda=0):
    """
    计算梯度和损失
    """
    #### Compute the forward pass
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    #隐藏层
    s1 = data.dot(W1) + b1
    h = sigmoid(s1)
    #输出层
    s2 = h.dot(W2) + b2
    y = softmax(s2)
    cost = - np.mean(labels * np.log(y))
    
    #判断有无正则项
    if Lambda != 0:
        cost += Lambda * (np.linalg.norm(W1) ** 2 + np.linalg.norm(W2) ** 2)
    
    
    #### Backward pass: compute gradients
    #第二层梯度
    N2, D2 = y.shape
    t2 = y - labels
    db2 = np.sum(t2, axis=0) / N2
    dW2 = h.T.dot(t2) / N2
    if Lambda != 0:
        dW2 += 2 * Lambda * W2
    dX2 = t2.dot(W2.T)
    
    #第一层梯度
    N2, D2 = data.shape
    t1 = h * (1 - h)
    dW1 = data.T.dot(t1 * dX2) / N2
    if Lambda != 0:
        dW1 += 2 * Lambda * W1
    db1 = np.sum(t1, axis=0) / N2
    
    grad = {}
    grad['W1'] = dW1
    grad['W2'] = dW2
    grad['b1'] = db1
    grad['b2'] = db2

    return cost, grad

def nn_train(trainData, trainLabels, devData, devLabels, 
             num_hidden = 300, learning_rate = 5, epoch=30, Lambda=0):
    (m, n) = trainData.shape
    d = trainLabels.shape[1]
    params = {}

    ### YOUR CODE HERE
    batch = 1000
    num = m // batch
    #epoch = 50
    #记录损失和正确率
    Cost = []
    Train_accuracy = []
    Dev_accuracy = []
    #初始化
    params['W1'] = np.random.randn(n, num_hidden)
    params['b1'] = np.zeros(num_hidden)
    params['W2'] = np.random.randn(num_hidden, d)
    params['b2'] = np.zeros(d)
    for _ in range(epoch):
        print(_)
        for i in range(num):
            #计算损失和梯度
            cost, grad = loss(trainData[i*batch: (i+1)*batch, :], trainLabels[i*batch: (i+1)*batch, :], params, Lambda=Lambda)
            #更新
            params['W1'] -= learning_rate * grad['W1']
            params['b1'] -= learning_rate * grad['b1']
            params['W2'] -= learning_rate * grad['W2']
            params['b2'] -= learning_rate * grad['b2']
            if i == num - 1:
                Cost.append(cost)
                Train_accuracy.append(nn_test(trainData, trainLabels, params))
                Dev_accuracy.append(nn_test(devData, devLabels, params))
    
    plt.plot(range(epoch), Cost)
    plt.title("loss VS epoch")
    plt.show()
    
    plt.plot(range(epoch), Train_accuracy, label="Train_accuracy")
    plt.plot(range(epoch), Dev_accuracy, label="Dev_accuracy")
    plt.show()
        
    ### END YOUR CODE

    return params

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

np.random.seed(100)
trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')
trainLabels = one_hot_labels(trainLabels)
p = np.random.permutation(60000)
trainData = trainData[p,:]
trainLabels = trainLabels[p,:]

devData = trainData[0:10000,:]
devLabels = trainLabels[0:10000,:]
trainData = trainData[10000:,:]
trainLabels = trainLabels[10000:,:]

mean = np.mean(trainData)
std = np.std(trainData)
trainData = (trainData - mean) / std
devData = (devData - mean) / std

testData, testLabels = readData('images_test.csv', 'labels_test.csv')
testLabels = one_hot_labels(testLabels)
testData = (testData - mean) / std

####(a)
params1 = nn_train(trainData, trainLabels, devData, devLabels)

####(b)
Lambda = 0.0001
params2 = nn_train(trainData, trainLabels, devData, devLabels, Lambda=Lambda)

####(c)
readyForTesting = True
if readyForTesting:
    accuracy1 = nn_test(testData, testLabels, params1)
    print('Test accuracy: %f' % accuracy1)
    
    accuracy2 = nn_test(testData, testLabels, params2)
    print('Test accuracy: %f' % accuracy2)
