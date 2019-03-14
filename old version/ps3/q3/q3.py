# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:27:42 2019

@author: qinzhen
"""

import numpy as np

X = np.genfromtxt("x.dat")
y = np.genfromtxt("y.dat")
theta_true = np.genfromtxt("theta.dat")

def l1l2(X, y, Lambda):
    #数据维度
    n, d = X.shape
    #设置阈值
    D = 1e-5
    #设置初始值
    theta = np.zeros(d)
    #记录上一轮迭代的theta
    theta_pre = np.copy(theta)
    while True:
        #坐标下降
        for i in range(d):
            #第i列
            Xi = X[:, i]
            #theta第i个元素为0
            theta[i] = 0
            #计算
            temp1 = X.dot(theta) - y
            temp2 = np.max([- (temp1.T.dot(Xi) + Lambda) / (Xi.T.dot(Xi)), 0])
            temp3 = np.min([- (temp1.T.dot(Xi) - Lambda) / (Xi.T.dot(Xi)), 0])
            #情形1
            theta[i] = temp2
            loss1 = 1 / 2 * np.sum((X.dot(theta) - y) ** 2) + Lambda * np.sum(np.abs(theta))
            #情形2
            theta[i] = temp3
            loss2 = 1 / 2 * np.sum((X.dot(theta) - y) ** 2) + Lambda * np.sum(np.abs(theta))
            
            #根据较小的loss对应的值更新
            if(loss1 < loss2):
                theta[i] = temp2
            else:
                theta[i] = temp3
        
        #计算误差
        delta = np.linalg.norm(theta - theta_pre)
        if delta < D:
            break

        theta_pre = np.copy(theta)
            
    return theta

theta = l1l2(X, y, 1)
print(theta)