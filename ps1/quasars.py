import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from __future__ import division

def load_data():
    train = np.genfromtxt('quasar_train.csv', skip_header=True, delimiter=',')
    test = np.genfromtxt('quasar_test.csv', skip_header=True, delimiter=',')
    wavelengths = np.genfromtxt('quasar_train.csv', skip_header=False, delimiter=',')[0]
    return train, test, wavelengths

def add_intercept(X_):
    X = None
    #####################
    

    ###################
    return X

def smooth_data(raw, wavelengths, tau):
    smooth = None
    ################

    ################
    return smooth

def LWR_smooth(spectrum, wavelengths, tau):
    smooth_spectrum = None
    ###############


    ###############
    return smooth_spectrum

def LR_smooth(Y, X_):
    X = add_intercept(X_)
    yhat = np.zeros(Y.shape)
    theta = np.zeros(2)
    #####################


    #####################
    return yhat, theta

def plot_b(X, raw_Y, Ys, desc, filename):
    plt.figure()
    ############


    ############
    plt.savefig(filename)

def plot_c(Yhat, Y, X, filename):
    plt.figure()
    ############


    #############
    plt.savefig(filename)
    return

def split(full, wavelengths):
    left, right = None, None
    ###############

    ###############
    return left, right

def dist(a, b):
    dist = 0
    ################


    ################
    return dist

def func_reg(left_train, right_train, right_test):
    m, n = left_train.shape
    lefthat = np.zeros(n)
    ###########################


    ###########################
    return lefthat

def main():
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
    pass


if __name__ == '__main__':
    main()
