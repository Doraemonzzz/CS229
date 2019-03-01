import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from __future__ import division


def load_data():
    X = np.genfromtxt('logistic_x.txt')
    Y = np.genfromtxt('logistic_y.txt')
    return X, Y

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))

    ################


    ################

    return X

def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    ##############


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


    ###########

    return loss

def calc_hessian(X, Y, theta):
    m, n = X.shape
    H = np.zeros((n, n))

    ##############


    #############

    return H

def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)

    ############


    ############

    return theta

def plot(X, Y, theta):
    plt.figure()

    ############


    ############

    plt.savefig('ps1q1c.png')
    return

def main():
    X_, Y = load_data()
    X = add_intercept(X_)
    theta = logistic_regression(X, Y)
    plot(X, Y, theta)

if __name__ == '__main__':
    main()
