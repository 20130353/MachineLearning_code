# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 9/29/18
# file: perceptron.py
# description:

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class perceptron():
    def __init__(self,max_iter,eta):
        self.max_iter = max_iter
        self.eta = eta
        self.dims = 0
        self.N = 0

    # SGD optimization
    def fit(self,x,y):
        self.N,self.dims = np.shape(x)
        self.w = np.ones((1,self.dims))
        self.b = 1.0

        for _ in range(self.max_iter):

            error_x = []
            error_y = []
            for inx,x_i in enumerate(x):
                y_pred = 1 if np.dot(self.w,x_i.transpose()) + self.b > 0 else 0
                if y_pred != y[inx]:
                   error_x.append(x_i)
                   error_y.append(y[inx])

            inx = np.random.permutation(np.shape(error_y)[0])
            gradient = np.dot(error_y[inx[0]],error_y[inx[0]])
            self.w += self.eta * gradient
            self.b += error_y[inx[0]]

    # BGD optimization
    def fit_1(self, x, y):
        self.N, self.dims = np.shape(x)
        self.w = np.ones((1, self.dims))
        self.b = 1.0

        for _ in range(self.max_iter):
            for inx, x_i in enumerate(x):
                y_pred = 1 if np.dot(self.w, x_i.transpose()) + self.b > 0 else 0
                if y_pred != y[inx]:
                    gradient = np.dot(x_i, y[inx])
                    self.w += self.eta * gradient
                    self.b += y[inx]


    def predict(self,x):
        y_pred = [1 if np.dot(self.w,x_i.transpose()) + self.b > 0 else 0 for x_i in x]
        return y_pred

if __name__ == '__main__':

    data = load_breast_cancer()
    x_train,x_test,y_train,y_test = train_test_split(data['data'],data['target'])

    model = perceptron(max_iter=100,eta=0.1)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(y_test)
    print(y_pred)
    print ('accuracy %f' %accuracy)

