# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:38:10 2019

@author: qinzhen
"""

import numpy as np

def load_data():
    X = np.loadtxt('data/x.dat')
    y = np.loadtxt('data/y.dat')
    
    return X, y