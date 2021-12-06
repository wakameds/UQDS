import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

def load_data(filename):
    """
    function to load data
    :param filename(str): file name
    :return: X(array):input vector, Y(array):output vector
    """
    X, Y = [], []
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            line = ' '.join(line.rstrip().split()).split(" ")
            if line[0] == "":
                continue
            X.append(line[:-1])
            Y.append(line[-1])

    return np.array(X, np.float32), np.array(Y, np.float32)

X_train, Y_train = load_data("housing_train.txt")
X_test, Y_test = load_data("housing_test.txt")

#Add bias to input
X_train = np.hstack([np.ones([X_train.shape[0], 1]), X_train])
X_test = np.hstack([np.ones([X_test.shape[0], 1]), X_test])


def compute_weight(x, y):
    """
    function to compute weight vector
    :param x(array): input vector
    :param y(array): output vector
    :return: w(array) weight vector  w = (xTx)-1xTy
    """
    return np.linalg.pinv(x).dot(y)


def regression(x, w):
    """
    function to predict y
    :param x(array): input vector
    :param w(array): weight vector
    :return: y_hat(array) output vector   x·wT
    """
    return x.dot(w)


def MSE(y_true, y_pred):
    """
    function to compute mean squared error
    :param y_true(array): y
    :param y_pred(array): y hat
    :return: MSE  1/n · (y-y_hat)^2
    """
    return np.mean((y_true-y_pred)**2)

def compute_polynomial(x, d):
    """
    function to make polynomial input vector
    :param x(array): input vector
    :param d(int): degree for polynomial
    :return: x_pol(array): polynomial input vector
    """
    X_polynominal = []
    for i in range(1, d+1):
        pol = np.power(x, i)
        if i == 1:
            X_polynominal = pol
        else:
            X_polynominal = np.concatenate((X_polynominal, pol), axis=1)
    return X_polynominal


#Regression model
W = compute_weight(X_train, Y_train)
MSE_train = MSE(Y_train, regression(X_train, W))
MSE_test = MSE(Y_test, regression(X_test, W))
print("train: {:.3f}, test: {:.3f}".format(MSE_train, MSE_test))



#Polynominal regression model
X_train_pol = compute_polynomial(X_train, 1)
X_test_pol = compute_polynomial(X_test, 1)


#Generalization error vs model complexity
errs_tr = []
errs_ts = []
max_degree = 5

for i in range(1,max_degree):
    X_train_pol = compute_polynomial(X_train, i)
    W_pol = compute_weight(X_train_pol, Y_train)

    errs_tr.append(MSE(Y_train, regression(X_train_pol, W_pol)))
    errs_ts.append(MSE(Y_test, regression(compute_polynomial(X_test_pol, i), W_pol)))

plt.plot(np.arange(1,max_degree), errs_tr, '-o', label='train')
plt.plot(np.arange(1,max_degree), errs_ts, '-o', label='test')
plt.legend()
plt.xlabel('degree $i$')
plt.ylabel('$Loss$')
plt.show()
