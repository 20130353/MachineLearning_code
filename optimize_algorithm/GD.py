# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 18-9-16
# file: GD.py
# description:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_regression

# batch gradient desent
# loss is 1/2m in order to simplify the calculation
def bgd(alpha, x, y, num_iter):
    # update the w with the whole samples
    m,n = x.shape  # number of samples
    theta = np.ones(n) # w
    J_list = []

    x_transpose = x.transpose()
    for iter in range(0, num_iter):
        hypothesis = np.dot(x, theta) # inner product: w*x
        loss = y - hypothesis # whole loss of whole samples
        J = np.sum(loss ** 2) / (2 * m)  # whole samples cost: sum[(yi-wi*xi)^2]/2m
        J_list.append(J)
        print("iter %s | J: %.3f" % (iter, J))

        gradient = np.dot(x_transpose, loss) / m # (y-w*x)*x]
        theta += alpha * gradient  # update w

    return theta,J_list


def sgd(alpha, x, y, num_iter):
    # update w with a single sample

    m,n = x.shape  # number of samples
    theta = np.ones(n) # w
    J_list = []

    # 随机化序列
    idx = np.random.permutation(y.shape[0])
    x, y = x[idx], y[idx]

    for j in range(num_iter):

        # in one interation, update theta w values m times
        for i in idx:
            single_hypothesis = np.dot(x[i], theta) # w*xi
            single_loss = y[i] - single_hypothesis # single loss: yi- w*xi
            gradient = np.dot(x[i].transpose(), single_loss) # [yi-w*xi]xi
            theta += alpha * gradient  # update

        # lasted whole loss
        hypothesis = np.dot(x, theta)
        loss = y - hypothesis
        J = np.sum(loss ** 2) / (2 * m)  # cost
        J_list.append(J)
        print("iter %s | J: %.3f" % (j, J))

    return theta, J_list


def mbgd(alpha, x, y, num_iter, minibatches):
    # update w with some samples
    m,n = x.shape  # number of samples
    theta = np.ones(n)
    J_list = []

    for j in range(num_iter):

        idx = np.random.permutation(y.shape[0])
        x, y = x[idx], y[idx]
        # split x/y into several subarray
        mini = np.array_split(range(y.shape[0]), minibatches)

        for i in mini:
            mb_hypothesis = np.dot(x[i], theta) #here, x[i] contains several samples
            mb_loss = y[i] - mb_hypothesis
            gradient = np.dot(x[i].transpose(), mb_loss) / minibatches
            theta += alpha * gradient  # update w

        # save loss
        hypothesis = np.dot(x, theta)
        loss = y - hypothesis
        J = np.sum(loss ** 2) / (2 * m)  # cost
        J_list.append(J)
        print("iter %s | J: %.3f" % (j, J))

    return theta, J_list

# based on the MBGD
def momentum(alpha, x, y, num_iter, minibatches, momentum):
    # history movement direction
    m, n = x.shape  # number of samples
    theta = np.ones(n)
    v = np.zeros(n) # same size with w
    J_list = []

    for j in range(num_iter):

        idx = np.random.permutation(y.shape[0])
        x, y = x[idx], y[idx]
        # split x/y into several subarray
        mini = np.array_split(range(y.shape[0]), minibatches)

        for i in mini:
            hypothesis = np.dot(x[i], theta)  # here, x[i] contains several samples
            loss = y[i] - hypothesis
            gradient = np.dot(x[i].transpose(), loss) / minibatches
            v = momentum * v + alpha * gradient
            theta += v  # update w

        # totoal loss
        hypothesis = np.dot(x, theta)
        loss = y - hypothesis
        J = np.sum(loss ** 2) / (2 * m)  # cost
        J_list.append(J)
        print("iter %s | J: %.3f" % (j, J))

    return theta, J_list


# based on the MBGD
def Nesterov(alpha, x, y, num_iter, minibatches, momentum):
    # update w by using the predicted point
    # history movement direction
    # first update w and then update v and w
    m, n = x.shape  # number of samples
    theta = np.ones(n)
    v = np.zeros(n)  # same size with w
    J_list = []

    for j in range(num_iter):

        idx = np.random.permutation(y.shape[0])
        x, y = x[idx], y[idx]
        # split x/y into several subarray
        mini = np.array_split(range(y.shape[0]), minibatches)

        for i in mini:
            theta += momentum * v
            hypothesis = np.dot(x[i], theta)  # here, x[i] contains several samples
            loss = y[i] - hypothesis
            gradient = np.dot(x[i].transpose(), loss) / minibatches
            v = momentum * v + alpha * gradient
            theta += v  # update w

        # totoal loss
        hypothesis = np.dot(x, theta)
        loss = y - hypothesis
        J = np.sum(loss ** 2) / (2 * m)  # cost
        J_list.append(J)
        print("iter %s | J: %.3f" % (j, J))

    return theta, J_list

# based on the BGD
def Adagrad(alpha, x, y, num_iter, gramma = 0.1):
    # gramma is smooth value
    # update the w with the whole samples
    m, n = x.shape  # number of samples
    theta = np.ones(n)  # w
    J_list = []
    G = 0.0

    x_transpose = x.transpose()
    for iter in range(0, num_iter):
        hypothesis = np.dot(x, theta)  # inner product: w*x
        loss = y - hypothesis  # whole loss of whole samples
        J = np.sum(loss ** 2) / (2 * m)  # whole samples cost: sum[(yi-wi*xi)^2]/2m
        J_list.append(J)
        print("iter %s | J: %.3f" % (iter, J))

        gradient = np.dot(x_transpose, loss) / m  # (y-w*x)*x]
        G += gradient
        theta += alpha/((G + gramma) ** 1/2) * gradient  # update w

    return theta, J_list

def RMSprop(alpha, x, y, num_iter, gramma = 0.1):

    # update the w with the whole samples
    m, n = x.shape  # number of samples
    theta = np.ones(n)  # w
    J_list = []
    G = 0.0 # history gradient values

    x_transpose = x.transpose()
    for iter in range(0, num_iter):
        hypothesis = np.dot(x, theta)  # inner product: w*x
        loss = y - hypothesis  # whole loss of whole samples
        J = np.sum(loss ** 2) / (2 * m)  # whole samples cost: sum[(yi-wi*xi)^2]/2m
        J_list.append(J)
        print("iter %s | J: %.3f" % (iter, J))

        gradient = np.dot(x_transpose, loss) / m  # (y-w*x)*x]
        G += gradient ** 2
        theta += alpha/np.sqrt((np.mean(G) + gramma)) * gradient  # update w

    return theta, J_list

def Adadelta(alpha, x, y, num_iter, beta=0.9,gramma = 0.1):
    # not finished ...

    return


if __name__ == '__main__':

    max_iter = 1000
    sample_size = 100
    feature_size = 2

    x, y = make_regression(n_samples=sample_size, n_features=feature_size, n_informative=1,
                           random_state=0, noise=35)
    m, n = np.shape(x)
    x = np.c_[np.ones(m), x]  # insert column, bias
    alpha = 0.01  # learning rate

    # pylab.plot(x[:, 1], y, 'r--')
    fig,ax = plt.subplots()


    print("\n#***BGD***#\n")
    theta_bgd,J_bgd = bgd(alpha, x, y, max_iter)
    ax.plot(J_bgd,color='k',linestyle='-')

    print("\n#***SGD***#\n")
    theta_sgd,J_sgd = sgd(alpha, x, y, max_iter)
    ax.plot(J_sgd, color='red', linestyle='-')

    print("\n#***MBGD***#\n")
    theta_mbgd,J_mbgd = mbgd(alpha, x, y, max_iter, 10)
    ax.plot(J_mbgd, color='darkgoldenrod', linestyle='-')

    print("\n#*** Momentum ***#\n")
    theta_momentum,J_momentum = momentum(alpha, x, y, max_iter, minibatches=10, momentum=0.9)
    ax.plot(J_momentum, color='blue', linestyle='-')

    print("\n#*** Nesterov ***#\n")
    theta_Nesterov,J_nesterov = Nesterov(alpha, x, y, max_iter, minibatches=10, momentum=0.9)
    ax.plot(J_nesterov, color='brown', linestyle='-')


    print("\n#*** Adagrad ***#\n")
    theta_Adagrad,J_adagrad = Adagrad(alpha, x, y, max_iter)
    ax.plot(J_adagrad, color='fuchsia', linestyle='-',linewidth='3')

    # RMS 和 Ada 颜色是混在一起的
    print("\n#***  RMSprop ***#\n")
    theta_RMSprop,J_RMSprop =  RMSprop(alpha, x, y, max_iter)
    ax.plot(J_RMSprop, color='green', linestyle='-',linewidth='2')

    plt.legend(['BGD','SGD','MBGD','Momentum','Nesterov','Adagrad','RMSprop'])

    plt.savefig('maxiter-'+str(max_iter)+'-sample-'+str(sample_size)+'-feature-'+str(feature_size)+'.tif')
    # plt.show()
    print("Done!")

