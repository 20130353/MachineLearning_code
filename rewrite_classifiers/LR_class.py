# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 18-9-14
# file: LR_class.py
# description:

# QA: why some people use the cross_entropy*prod to update w value

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class LogisticRegression():

    def __init__(self,max_iter):

        self.eta = 0.1
        self.max_iter = max_iter
        self.w = None

    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))

    def fit(self,x,y):
        x = np.mat(x)
        y = np.mat(y).transpose()
        m, n = np.shape(x)
        w = np.ones((n, 1))

        for k in range(self.max_iter):
            for i in range(m):
                # sigmoid operating on the summation of whole diemensions
                y_pre = self.sigmoid(x*w)
                grad = x.transpose()*(y[i] - y_pre)
                w = w + self.eta * grad
        self.w = w

    def predict(self,x):
        return [np.sign(np.array(self.sigmoid(each*self.w))[0]) for each in x]


if __name__ == '__main__':

    data = load_breast_cancer()
    x_train,x_test,y_train,y_test = train_test_split(data['data'],data['target'])
    model = LogisticRegression(max_iter=100)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ('accuracy %f' % accuracy)