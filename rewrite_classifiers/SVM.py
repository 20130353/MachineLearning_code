# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 9/30/18
# file: SVM.py
# description: init version for SVM classifier and accuracy is 0.34, maybe there is something wrong!

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import math
from kernels import *


class SVM():

    def __init__(self,kernel,max_iter,C,difference=1e-3):
        self.kernel = kernel
        if kernel == 'linear':
            self._kernel = LinearKernel()
        elif kernel == 'gauss':
            self._kernel = RBF()

        self.max_iter = max_iter
        self.alpha = []
        self.C = C
        self.N = 0
        self.dims = 0
        self.w = np.array(self.dims)
        self.b = 0.0
        self.diff = difference
        self.K = None


    def _rand_index(self,i):
        j = i
        while(j == i):
            j = np.random.randint(0,self.N-1)
        return j

    def _cal_error(self,x,y):
        return y - self.predict(x)

    def predict(self, x_test):
        if len(x_test.shape) == 1: # one dims data
            k = self._kernel(self.x, x_test)
            y_pred = np.dot((self.alpha * self.y), k) + self.b
            return y_pred
        else:
            result = []
            for inx in range(x_test.shape[0]):
                k = self._kernel(self.x,x_test[inx])
                y_pred = np.dot((self.alpha * self.y), k) + self.b
                result.append(np.sign(y_pred))  # 正的返回1，负的返回-1
            return result

    def get_bounds(self, i, j):

        #if y1 != y2: L = max(0,old_a2-old_a1) H = min(C, C+old_a2-old_a1)
        #if y1 == y2: L = max(0,old_a2-old_a1-C) H = min(C, old_a2+old_a1)
        # L = max(0,old_a2-old_a1) H = min(C, C+old_a2-old_a1)
        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[j] + self.alpha[i] - self.C)
            H = min(self.C, self.alpha[j] + self.alpha[i])

        return L, H

    def _kernel(self,v1,v2):
        if self.kernel == 'line':
            return np.multiply(v1,v2)
        elif self.kernel[0] == 'gauss':
            k = sum((v1[inx]-v2[inx])**2 for inx in range(self.dims))
            return math.exp(-0.5 * K / (self.sigma ** 2))


    def fit(self,x,y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.N,self.dims = np.shape(x)
        self.alpha = np.zeros(self.N)
        self.K = np.zeros((self.N,self.N))

        for i in range(self.N):
            self.K[:, i] = self._kernel(self.x, self.x[i, :])  #innerj-product kernel values

        for _ in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)

            for j in range(self.N): # calculate alpha_i and alpha_j, fix the alpha_i

                i = self._rand_index(j)
                E_i,E_j = self._cal_error(x[i],y[i]), self._cal_error(x[j],y[j])

                # judge whethe statisfy the KTT condition
                if (self.y[j] * E_j < -0.001 and self.alpha[j] < self.C) or (self.y[j] * E_j > 0.001 and self.alpha[j] > 0):

                    # update w
                    L, H = self.getBounds(i, j)
                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j] # step length
                    if eta == 0: # step length is 0 and stop the process
                        continue
                    alpha_j_old,alpha_i_old = self.alpha[j],self.alpha[i]
                    self.alpha[j] -= (self.y[j] * (E_i - E_j))/ eta  # alpha_2_new = alpha_2_old - y2(e1-e2)/eta
                    self.alpha[j] = min(H,self.alpha[j]) if self.alpha[j] > L else max(L,self.alpha[j]) # check alpha_j is in [L,H]

                    self.alpha[i] = alpha_i_old + (self.y[i] * self.y[j]) * (alpha_j_old - self.alpha[i]) # alpha_1_new = alpha_1_old + y1y2(alpha_2_old-alpha_1_old)

                    # update b
                    b1 = self.b - E_i - self.y[i] * (self.alpha[i] - alpha_j_old) * self.K[i, i] - \
                         self.y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
                    b2 = self.b - E_j - self.y[j] * (self.alpha[j] - alpha__j_old) * self.K[j, j] - \
                         self.y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = 0.5 * (b1 + b2)

                # judge convergence
                diff = np.linalg.norm(self.alpha - alpha_prev)
                if diff < self.diff:
                    break


if __name__ == '__main__':

    data = load_breast_cancer()
    x_train,x_test,y_train,y_test = train_test_split(data['data'],data['target'])
    model = SVM(max_iter=100,kernel='gauss',C=1)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ('accuracy %f' % accuracy)
