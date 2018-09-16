# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 18-9-14
# file: LR_class.py
# description:

# QA: why some people use the cross_entropy*prod to update w value


import numpy as np

class LR_class():

    def __init__(self):

        self.eta = 0.1
        self.penalty = 'L2'
        self.interation = 500
        self.w = None


    def sigmoid(self,x):
        return 1.0/(1+np.exp(-x))

    def fit(self,x,y):
        x = np.mat(x)
        y = np.mat(y).transpose()
        m, n = np.shape(x)
        w = np.ones((n, 1))

        for k in range(self.interation):
            for i in range(m):
                # sigmoid operating on the summation of whole diemensions
                y_pre = self.sigmoid(x*w)
                grad = x.transpose()*(y[i] - y_pre)
                w = w + self.eta * grad
        self.w = w

    def predict(self,x):
        y_pres = []
        for each in x:
            y = np.float(np.array(self.sigmoid(each*self.w))[0])
            if y > 0.5:
                y_pres.append(1)
            else:
                y_pres.append(-1)

        return y_pres
