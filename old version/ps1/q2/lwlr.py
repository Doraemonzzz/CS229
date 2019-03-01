# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:43:04 2019

@author: qinzhen
"""
import numpy as np

#定义h(theta, X)
def h(theta, X):
    return 1 / (1 + np.exp(- X.dot(theta)))

tau = 1
Lambda = 0.0001
threshold = 1e-6

def lwlr(X_train, y_train, x, tau):
    #记录数据维度
    m, d = X_train.shape
    #初始化
    theta = np.zeros(d)
    #计算权重
    norm = np.sum((X_train - x) ** 2, axis=1)
    W = np.exp(- norm / (2 * tau ** 2))
    #初始化梯度
    g = np.ones(d)
    
    while np.linalg.norm(g) > threshold:
        #计算h(theta, X)
        h_X = h(theta, X_train)
        #梯度
        z = W * (y_train - h_X)
        g = X_train.T.dot(z) - Lambda * theta
        #Hessian矩阵
        D = - np.diag(W * h_X * (1 - h_X))
        H = X_train.T.dot(D).dot(X_train) - Lambda * np.eye(d)
        
        #更新
        theta -= np.linalg.inv(H).dot(g)
    
    ans = (theta.dot(x) > 0).astype(np.float64)
    return ans


