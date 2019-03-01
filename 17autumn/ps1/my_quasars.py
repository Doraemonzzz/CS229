# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 21:08:25 2018

@author: Administrator
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from numpy.linalg import inv
#from __future__ import division

def load_data():
    train = np.genfromtxt('quasar_train.csv', skip_header=True, delimiter=',')
    test = np.genfromtxt('quasar_test.csv', skip_header=True, delimiter=',')
    wavelengths = np.genfromtxt('quasar_train.csv', skip_header=False, delimiter=',')[0]
    return train, test, wavelengths

def add_intercept(X_):
    X = None
    #####################
    X_ = X_.reshape((-1, 1))
    m, n = X_.shape
    ones = np.ones((m, 1))
    X = np.append(ones, X_, axis = 1)

    ###################
    return X

def smooth_data(raw, wavelengths, tau):
    smooth = None
    ################
    smooth = []
    for spectrum in raw:
        smooth.append(LWR_smooth(spectrum, wavelengths, tau))
    ################
    return np.array(smooth)

def LWR_smooth(spectrum, wavelengths, tau):
    smooth_spectrum = np.array([])
    ###############
    X = add_intercept(wavelengths)
    Y = spectrum.reshape((-1, 1))
    for i in range(len(wavelengths)):
        w = np.exp(-(wavelengths - wavelengths[i])**2 / (2*tau**2))
        W = np.diag(w)
        theta = inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(Y)
        yhat = theta.T.dot(X[i])
        smooth_spectrum = np.append(smooth_spectrum, yhat)
    ###############
    return smooth_spectrum

#利用最小二乘公式
def LR_smooth(Y, X_):
    X = add_intercept(X_)
    Y = Y.reshape((-1, 1))
    yhat = np.zeros(Y.shape)
    theta = np.zeros(2)
    #####################
    theta = inv(X.T.dot(X)).dot(X.T).dot(Y)
    yhat = X.dot(theta)

    #####################
    return yhat, theta

def plot_b(X, raw_Y, Ys, desc, filename):
    plt.figure()
    ############
    for i in range(len(Ys)):
        plt.scatter(X, raw_Y, c='red', s=2, label='raw_data')
        plt.plot(X, Ys[i], c='blue', label='Regression line')
        plt.title(desc[i])
        plt.show()

    ############
    #plt.savefig(filename)
    
#将左边右边区分开来,左边<1200,右边>=1300
def plot_c(Yhat, Y, X, filename):
    plt.figure()
    ############
    plt.plot(X[:50],Yhat)
    plt.plot(X,Y)
    plt.show()
    #############
    plt.savefig(filename)
    return

def split(full, wavelengths):
    left, right = None, None
    ###############
    indexl = np.argwhere(wavelengths == 1200)[0][0]
    indexr = np.argwhere(wavelengths == 1300)[0][0]
    left = full[:, :indexl]
    right = full[:, indexr:]
    ###############
    return left, right

def dist(a, b):
    dist = 0
    ################
    dist = ((a - b)**2).sum()
    ################
    return dist

def func_reg(left_train, right_train, right_test):
    m, n = left_train.shape
    #m=200,n=50
    lefthat = np.zeros(n)
    ###########################
    #right_train 200*300
    #right_test  1*300
    #left_train  200*50
    #求题目中的d(f1,f2),先求每个点的距离,200*300矩阵
    d = (right_train - right_test)**2
    #按照行求和200*1
    d1 = d.sum(axis=1)
    #找到排名前3的作为neighb_k(f_right)
    tempd = d1.copy()
    tempd.sort()
    #找到索引
    index = (d1==tempd[0])|(d1==tempd[1])|(d1==tempd[2])
    #h为d1中的最大值
    h = d1.max()
    d1 = d1/h
    #ker (1-t)
    d1 = 1 - d1
    #求lefthat
    a = (d1[index].dot(left_train[index]))
    b = d1[index].sum()
    
    lefthat = a/b
    ###########################
    return lefthat


raw_train, raw_test, wavelengths = load_data()

## Part b.i
lr_est, theta = LR_smooth(raw_train[0], wavelengths)
print('Part b.i) Theta=[%.4f, %.4f]' % (theta[0], theta[1]))
plot_b(wavelengths, raw_train[0], [lr_est], ['Regression line'], 'ps1q5b1.png')

## Part b.ii
lwr_est_5 = LWR_smooth(raw_train[0], wavelengths, 5)
plot_b(wavelengths, raw_train[0], [lwr_est_5], ['tau = 5'], 'ps1q5b2.png')

### Part b.iii
lwr_est_1 = LWR_smooth(raw_train[0], wavelengths, 1)
lwr_est_10 = LWR_smooth(raw_train[0], wavelengths, 10)
lwr_est_100 = LWR_smooth(raw_train[0], wavelengths, 100)
lwr_est_1000 = LWR_smooth(raw_train[0], wavelengths, 1000)
plot_b(wavelengths, raw_train[0],
         [lwr_est_1, lwr_est_5, lwr_est_10, lwr_est_100, lwr_est_1000],
         ['tau = 1', 'tau = 5', 'tau = 10', 'tau = 100', 'tau = 1000'],
         'ps1q5b3.png')

### Part c.i
smooth_train, smooth_test = [smooth_data(raw, wavelengths, 5) for raw in [raw_train, raw_test]]

#### Part c.ii
left_train, right_train = split(smooth_train, wavelengths)
left_test, right_test = split(smooth_test, wavelengths)

train_errors = [dist(left, func_reg(left_train, right_train, right)) for left, right in zip(left_train, right_train)]
print('Part c.ii) Training error: %.4f' % np.mean(train_errors))

### Part c.iii
test_errors = [dist(left, func_reg(left_train, right_train, right)) for left, right in zip(left_test, right_test)]
print('Part c.iii) Test error: %.4f' % np.mean(test_errors))

left_1 = func_reg(left_train, right_train, right_test[0])
plot_c(left_1, smooth_test[0], wavelengths, 'ps1q5c3_1.png')
left_6 = func_reg(left_train, right_train, right_test[5])
plot_c(left_6, smooth_test[5], wavelengths, 'ps1q5c3_6.png')

