# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 16:12:53 2019

@author: qinzhen
"""
import numpy as np
import matplotlib.pyplot as plt

Lambda = 0.0001
threshold = 1e-6

#读取数据
def load_data():
    X = np.loadtxt('data/x.dat')
    y = np.loadtxt('data/y.dat')
    
    return X, y

#定义h(theta, X)
def h(theta, X):
    return 1 / (1 + np.exp(- X.dot(theta)))

#计算
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

#作图
def plot_lwlr(X, y, tau):
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    d = xx.ravel().shape[0]
    Z = np.zeros(d)
    data = np.c_[xx.ravel(), yy.ravel()]
    
    for i in range(d):
        x = data[i, :]
        Z[i] = lwlr(X, y, x, tau)
    
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    X0 = X[y == 0]
    X1 = X[y == 1]
    plt.scatter(X0[:, 0], X0[:, 1], marker='x')
    plt.scatter(X1[:, 0], X1[:, 1], marker='o')
    plt.title("tau="+str(tau))
    plt.show()

Tau = [0.01, 0.05, 0.1, 0.5, 1, 5]
X, y = load_data()
for tau in Tau:
    plot_lwlr(X, y, tau)