# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 18-9-14
# file: Perceptron_class.py
# description:

import numpy as np

class Perceptron_class():

    def __init__(self):
        self.eta = 0.1
        self.interation = 100

    def fit(self, x, y):
        m, n = np.shape(x)
        self.w = np.ones(n)
        self.b = np.ones(n)


        for i in range(self.interation):
            for j in range(m):
                if (sum(self.w * x[j] + self.b)) >= 0:
                    yhat = 1
                else:
                    yhat = -1

                if yhat != y[j]:
                    self.w = self.w + self.eta * sum(x[i] * y[j])
                    self.b = self.b + self.eta * y[j]
                    print(self.w)
                    print(self.b)

    def predict(self,x):
        y_pres = []
        for each in x:
            y = sum(each * self.w + self.b)
            if y > 0:
                y_pres.append(1)
            else:
                y_pres.append(-1)

        return y_pres

