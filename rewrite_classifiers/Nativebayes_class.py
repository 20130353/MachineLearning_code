# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 18-9-16
# file: Nativebayes_class.py
# description:this file is model about the nativebayes with Gaussian distribution

import numpy as np
from math import sqrt

class Nativebayes_class:

    def __init__(self):
        self.pre_prob = []
        self.cond_prob = []
        self.x = []
        self.y = []
        self.classes = []
        self.alpha = 1

    def pre_problity(self):
        pos_prob = 1.0 * (np.sum(self.y == 1.0) + self.alpha) / (self.N + self.alpha)
        neg_prob = 1.0 * (np.sum(self.y == 0.0) + 1) / (self.N + self.alpha)
        return [pos_prob, neg_prob]

    def fit(self,x,y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.N,self.dims = np.shape(x)
        self.pre_prob = self.pre_problity()


    def predict(self, test_x):

        test_x = np.array(test_x)
        res = []
        for k in range(len(test_x)):
            cond_result = np.zeros([self.N, self.dims])
            pos_data = self.x[self.y == 1.0]
            neg_data = self.x[self.y == 0.0]

            for i in range(self.dims):
                cond_result[i, 0] = 1.0 * (np.sum(pos_data[:, i] == test_x[0][i]) + self.alpha) / (
                        np.sum(self.y == 1.0) + self.alpha)
                cond_result[i, 1] = 1.0 * (np.sum(neg_data[:, i] == test_x[0][i]) + 1) / (
                        np.sum(self.y == 0.0) + self.alpha)

            for j in range(self.dims):
                # mean,std computation
                pos_mean = np.mean(self.x[(self.y == 1.0), j])
                pos_std = np.std(self.x[(self.y == 1.0), j])

                neg_mean = np.mean(self.x[(self.y == 0.0), j])
                neg_std = np.std(self.x[(self.y == 0.0), j])

                cond_result[j, 0] = 1.0 / (sqrt(2 * np.pi) * pos_std) * np.exp(
                    -1 * (test_x[0, j] - pos_mean) ** 2 / (2 * pos_std ** 2))
                cond_result[j, 1] = 1.0 / (sqrt(2 * np.pi) * neg_std) * np.exp(
                    -1 * (test_x[0, j] - neg_mean) ** 2 / (2 * neg_std ** 2))


            pos_result = self.pre_prob[0]
            neg_result = self.pre_prob[1]
            for i in range(np.shape(cond_result)[0]):
                pos_result *= cond_result[i, 0]
                neg_result *= cond_result[i, 1]

            if pos_result > neg_result:
                res.append(1)
            else:
                res.append(0)

        return res