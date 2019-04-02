# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:49:57 2019

@author: qinzhen
"""

### Independent Components Analysis
###
### This program requires a working installation of:
###
### On Mac:
###     1. portaudio: On Mac: brew install portaudio
###     2. sounddevice: pip install sounddevice
###
### On windows:
###      pip install pyaudio sounddevice
###

import sounddevice as sd
import numpy as np

Fs = 11025

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('mix.dat')
    return mix

def play(vec):
    sd.play(vec, Fs, blocking=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    ######## Your code here ##########
    
    for alpha in anneal:
        #打乱数据
        np.random.permutation(X)
        for i in range(M):
            x = X[i, :].reshape(-1, 1)
            WX = W.dot(x)
            grad = (1 - 2 * sigmoid(WX)).dot(x.T) + np.linalg.inv(W.T)
            W += alpha * grad
    ###################################
    return W

def unmix(X, W):
    S = np.zeros(X.shape)

    ######### Your code here ##########
    S = X.dot(W.T)
    ##################################
    return S

X = normalize(load_data())

for i in range(X.shape[1]):
    print('Playing mixed track %d' % i)
    play(X[:, i])

W = unmixer(X)
S = normalize(unmix(X, W))

for i in range(S.shape[1]):
    print('Playing separated track %d' % i)
    play(S[:, i])