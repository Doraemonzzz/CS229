# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 20:27:14 2018

@author: Administrator
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import inv
#from __future__ import division


def load_data():
    X = np.genfromtxt('logistic_x.txt')
    Y = np.genfromtxt('logistic_y.txt')
    return X, Y

#增加截距项
def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    ################
    ones = np.ones((m, 1))
    X = np.append(ones, X_, axis = 1)
    ################
    return X

#利用之前所述的公式计算
def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    ##############
    Y_ = Y.reshape([-1, 1])
    d1 = (X*Y_).dot(theta)
    h = 1 / (1 + np.exp(-d1))
    S = (1 - h) * Y
    grad = -1/m * (X.T).dot(S)

    ##############

    return grad

##
## This function is useful to debug
## Ensure that loss is going down over iterations
##
def calc_loss(X, Y, theta):
    m, n = X.shape
    loss = 0.

    ###########
    Y = Y.reshape([-1, 1])
    d1 = (X*Y).dot(theta)
    h = 1 / (1 + np.exp(-d1))
    loss = -1/m * np.sum(np.log(h))

    ###########

    return loss

def calc_hessian(X, Y, theta):
    m, n = X.shape
    H = np.zeros((n, n))

    ##############
    Y = Y.reshape([-1, 1])
    d1 = (X*Y).dot(theta)
    h = 1 / (1 + np.exp(-d1))
    S = np.diag(h * (1-h))
    H = X.T.dot(S).dot(X)

    #############

    return H

def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)

    ############
    H = calc_hessian(X, Y, theta)
    grad = calc_grad(X, Y, theta)
    theta -= inv(H).dot(grad)
    ############

    return theta

def plot(X, Y, theta):
    plt.figure()

    ############
    x1 = X[Y>0][:, 1]
    y1 = X[Y>0][:, 2]
    x2 = X[Y<0][:, 1]
    y2 = X[Y<0][:, 2]
    #计算系数
    theta = logistic_regression(X, Y)
    Min = np.min(X[:, 1])
    Max = np.max(X[:, 1])
    x = np.array([Min, Max])
    y = -(theta[0] + theta[1]*x)/theta[2]
    plt.scatter(x1, y1)
    plt.scatter(x2, y2)
    plt.plot(x, y)
    plt.title('Newton’s method for Logistic regression')

    ############

    plt.savefig('ps1q1c.png')
    return

#def main():
X_, Y = load_data()
X = add_intercept(X_)
theta = logistic_regression(X, Y)
plot(X, Y, theta)

#if __name__ == '__main__':
#    main()
