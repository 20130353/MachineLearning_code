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
    def __init__(self,max_iter,eta,batch_size=10):
        self.max_iter = max_iter
        self.eta = eta
        self.dims = 0
        self.N = 0
        self.batch_size = batch_size
        self.gramma = 0.9

    # SGD optimization
    def fit(self,x,y,gredient):
        self.N,self.dims = np.shape(x)
        self.w = np.zeros((1,self.dims))
        self.b = 0.0
        self.x = np.array(x)
        self.y = np.array(y)


        if gredient == 'BGD':
            self._BGD()
        elif gredient == 'SGD':
            self._SGD()
        elif gredient == 'MGD':
            self._MBGD()
        elif gredient == 'Momentum':
            self._Momentum()


    def _BGD(self):
        for _ in range(self.max_iter):
            x_error = []
            y_error = []

            # find all error samples for grandient
            for inx, x_i in enumerate(self.x):
                y_pred = 1 if sum(np.dot(self.w, x_i.transpose()) + self.b) > 0 else 0
                if y_pred != self.y[inx]:
                    x_error.append(x_i)
                    y_error.append(self.y[inx])

            gradient = np.dot(np.array(x_error).transpose(),y_error)
            self.w += self.eta * gradient
            self.b += self.eta * sum(np.array(y_error))


    # BGD optimization
    def _SGD(self):

        for _ in range(self.max_iter):

            inx = np.random.permutation(self.N)
            x, y = self.x[inx], self.y[inx]

            for inx, x_i in enumerate(x):
                y_pred = 1 if sum(np.dot(self.w, x_i.transpose())) + self.b > 0 else 0
                if y_pred != y[inx]:
                    gradient = np.dot(x_i, y[inx])
                    self.w += self.eta * gradient
                    self.b += self.eta * y[inx]

    def _MBGD(self):

        for _ in range(self.max_iter):

            inx = np.random.permutation(self.N)
            x, y = self.x[inx], self.y[inx]
            batches = np.array_split(range(self.N),self.batch_size)

            for batch in batches:
                x_batch,y_batch = x[batch],y[batch]

                x_error = []
                y_error = []

                # find all error samples for grandient
                for inx, x_i in enumerate(x_batch):
                    y_pred = 1 if sum(np.dot(self.w, x_i.transpose()) + self.b) > 0 else 0
                    if y_pred != y_batch[inx]:
                        x_error.append(x_i)
                        y_error.append(y_batch[inx])

                gradient = np.dot(np.array(x_error).transpose(), y_error)
                self.w += self.eta * gradient
                self.b += self.eta * sum(np.array(y_error))

    def _Momentum(self):

        for _ in range(self.max_iter):

            inx = np.random.permutation(self.N)
            x, y = self.x[inx], self.y[inx]
            batches = np.array_split(range(self.N),self.batch_size)
            v = 0.0

            for batch in batches:
                x_batch,y_batch = x[batch],y[batch]

                x_error = []
                y_error = []

                # find all error samples for grandient
                for inx, x_i in enumerate(x_batch):
                    y_pred = 1 if sum(np.dot(self.w, x_i.transpose()) + self.b) > 0 else 0
                    if y_pred != y_batch[inx]:
                        x_error.append(x_i)
                        y_error.append(y_batch[inx])

                gradient = np.dot(np.array(x_error).transpose(), y_error)
                v = self.gramma * v + self.eta * gradient
                self.w += v
                self.b += self.eta * sum(np.array(y_error))



    def predict(self, x):
        self.x_test = np.array(x)
        y_pred = [1 if np.dot(self.w,x_i.transpose()) + self.b > 0 else 0 for x_i in self.x_test]
        return y_pred

if __name__ == '__main__':

    data = load_breast_cancer()
    x_train,x_test,y_train,y_test = train_test_split(data['data'],data['target'])

    model = perceptron(max_iter=1, eta=1)
    model.fit(x_train, y_train, gredient='Momentum')
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print ('Momentum accuracy %f' % accuracy)

    model = perceptron(max_iter=1,eta=1)
    model.fit(x_train,y_train,gredient='MGD')
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print ('MGD accuracy %f' %accuracy)

