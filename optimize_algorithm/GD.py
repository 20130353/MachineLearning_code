# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 18-9-16
# file: GD.py
# description:


import numpy as np
import pylab
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

    pylab.plot(range(num_iter), J_list, "k-")
    return theta


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

    pylab.plot(range(num_iter), J_list, color='coral',linestyle = '-')
    return theta


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

    pylab.plot(range(num_iter), J_list, "y-")
    return theta

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

    pylab.plot(range(num_iter), J_list, "b-")
    return theta


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

    pylab.plot(range(num_iter), J_list, color = 'hotpink', linestyle = '-')
    return theta

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


    pylab.plot(range(num_iter), J_list, color = 'darkviolet', linestyle = '-')

    return theta

def RMSprop(alpha, x, y, num_iter, beta=0.9,gramma = 0.1):

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


    pylab.plot(range(num_iter), J_list, color = 'darksalmon', linestyle = '-')

    return theta

def Adadelta(alpha, x, y, num_iter, beta=0.9,gramma = 0.1):
    # not finished ...

    # # update the w with the whole samples
    # m, n = x.shape  # number of samples
    # theta = np.ones(n)  # w
    # J_list = []
    # G = 0.0 # history gradient values
    #
    # x_transpose = x.transpose()
    # for iter in range(0, num_iter):
    #     hypothesis = np.dot(x, theta)  # inner product: w*x
    #     loss = y - hypothesis  # whole loss of whole samples
    #     J = np.sum(loss ** 2) / (2 * m)  # whole samples cost: sum[(yi-wi*xi)^2]/2m
    #     J_list.append(J)
    #     print("iter %s | J: %.3f" % (iter, J))
    #
    #     gradient = np.dot(x_transpose, loss) / m  # (y-w*x)*x]
    #     G += gradient ** 2
    #
    #     gap_theta = alpha/np.sqrt((np.mean(G) + gramma)) * gradient
    #     expected_theta = beta * np.mean(G) + (1-beta)* (gap_theta ** 2)
    #     RMS_t = np.sqrt(expected_theta + gramma)
    #
    #     gap_theta = alpha / np.sqrt((np.mean(G[:,-1]) + gramma)) * gradient
    #     expected_theta = beta * np.mean(G) + (1 - beta) * (gap_theta ** 2)
    #     RMS_last = np.sqrt(expected_theta + gramma)
    #
    #     theta +=
    #
    #
    # pylab.plot(range(num_iter), J_list, color = 'darksalmon', linestyle = ':')

    return


if __name__ == '__main__':

    x, y = make_regression(n_samples=100, n_features=2, n_informative=1,
                           random_state=0, noise=35)
    m, n = np.shape(x)
    x = np.c_[np.ones(m), x]  # insert column, bias
    alpha = 0.01  # learning rate

    pylab.plot(x[:, 1], y, 'r--')

    print("\n#***BGD***#\n")
    theta_bgd = bgd(alpha, x, y, 100)
    # for i in range(x.shape[1]):
    #     y_bgd_predict = theta_bgd * x
    # pylab.plot(x, y_bgd_predict, 'k--')

    print("\n#***SGD***#\n")
    theta_sgd = sgd(alpha, x, y, 100)
    # for i in range(x.shape[1]):
    #     y_sgd_predict = theta_sgd * x
    # pylab.plot(x, y_sgd_predict, color = 'coral',linestyle='--')

    print("\n#***MBGD***#\n")
    theta_mbgd = mbgd(alpha, x, y, 100, 10)
    # for i in range(x.shape[1]):
    #     y_mbgd_predict = theta_mbgd * x
    # pylab.plot(x, y_mbgd_predict, 'y--')


    print("\n#*** Momentum ***#\n")
    theta_momentum = momentum(alpha, x, y, 100, 10, 0.9)
    # for i in range(x.shape[1]):
    #     y_momentum_predict = theta_momentum * x
    # pylab.plot(x, y_momentum_predict, 'b--')

    print("\n#*** Nesterov ***#\n")
    theta_Nesterov = Nesterov(alpha, x, y, 100, 10, 0.9)

    print("\n#*** Adagrad ***#\n")
    theta_Adagrad = Adagrad(alpha, x, y, 100)

    print("\n#***  RMSprop ***#\n")
    theta_RMSprop =  RMSprop(alpha, x, y, 100)


    pylab.legend(['BGD','SGD','MBGD','Momentum','Nesterov','Adagrad','RMSprop'])


    pylab.show()
    print("Done!")
