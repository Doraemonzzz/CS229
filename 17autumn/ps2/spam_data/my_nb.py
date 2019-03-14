# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:58:26 2019

@author: qinzhen
"""

import numpy as np
import matplotlib.pyplot as plt

def readMatrix(file):
    fd = open(file, 'r')
    #读取第一行
    hdr = fd.readline()
    #读取行列
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    #读取单词
    tokens = fd.readline().strip().split()
    #构造空矩阵
    matrix = np.zeros((rows, cols))
    Y = []
    #line为每行的元素
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        #第一个元素表示是否为垃圾邮件
        Y.append(nums[0])
        #将后续数据读入
        kv = np.array(nums[1:])
        #从第一个开始每两个累加
        k = np.cumsum(kv[:-1:2])
        #从第二个开始每隔一个取出
        v = kv[1::2]
        #这里应该是一种特殊的存储格式，我们直接使用即可
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)

def nb_train(matrix, category):
    #matrix(i,j)表示第i个邮件中第j个元素出现了几次
    state = {}
    #token的数量
    N = matrix.shape[1]
    #邮件数量
    M = matrix.shape[0]
    ###################
    
    #垃圾邮件的数量
    y1 = matrix[category==1]
    n1 = np.sum(y1)
    #非垃圾邮件的数量
    y0 = matrix[category==0]
    n0 = np.sum(y0)
    
    #P(y=1)
    p1 = y1.shape[0] / M
    #P(y=0)
    p0 = y0.shape[0] / M
    state[-1] = [p0, p1]
    
    for i in range(N):
        #找到第i个token
        #垃圾邮件中第i个token出现的数量
        s1 = matrix[category==1][:, i]
        #拉普拉斯平滑
        u1 = (s1.sum() + 1) / (n1 + N)
        #非垃圾邮件中第i个token出现的数量
        s0 = matrix[category==0][:, i]
        #拉普拉斯平滑
        u0 = (s0.sum() + 1) / (n0 + N)
        #存入字典
        state[i] = [u0, u1]

    ###################
    return state

def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    #邮件数量
    M = matrix.shape[0]
    #token的数量
    N = matrix.shape[1]
    for i in range(M):
        #第i个邮件
        vector = matrix[i]
        s1 = np.log(state[-1][1])
        s0 = np.log(state[-1][0])
        
        for j in range(N):
            #对第j个token的对数概率做累加
            s1 += vector[j] * np.log(state[j][1])
            s0 += vector[j] * np.log(state[j][0])
        if s1 > s0:
            output[i] = 1

    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)
    return error
    
def nb(file):
    trainMatrix, tokenlist, trainCategory = readMatrix(file)
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')
    
    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)
    
    return evaluate(output, testCategory)

trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

state = nb_train(trainMatrix, trainCategory)
output = nb_test(testMatrix, state)

evaluate(output, testCategory)

#problem b
b=[]
for i in range(1448):
    b.append((i,np.log(state[i][1])-np.log(state[i][0])))
    
b.sort(key=lambda i:i[-1],reverse=True)
key = b[:5]

word = []
for i in key:
    word.append(tokenlist[i[0]])
    
print(word)

#problem c
size = ['.50','.100','.200','.400','.800','.1400']
size1 = [50, 100, 200, 400, 800, 1400]
train = "MATRIX.TRAIN"
error = []
for i in size:
    file = train+i
    error.append(nb(file))
    
plt.plot(size, error)
plt.title("error VS szie")
