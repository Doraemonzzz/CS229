# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:12:37 2019

@author: qinzhen
"""

import matplotlib.pyplot as plt
import numpy as np
from lwlr import lwlr

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