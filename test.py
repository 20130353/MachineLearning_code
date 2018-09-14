# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 18-9-14
# file: test.py
# description: test the classification method waired by me

import numpy as np

def load_data(filename):
    x = []
    y = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split()
        x.append([1.0, float(line_arr[0]), float(line_arr[1])])
        y.append(int(line_arr[2]))
    return x, y

def accuracy_calculation(y, yhat):

    acc = [0 if y[inx] == yhat[inx] else 1 for inx in range(np.shape(y)[0])]
    return sum(acc)/np.shape(y)[0]

def LR_regression(x,y,tx):
    from sklearn.linear_model import LogisticRegression
    res = LogisticRegression().fit(x, y).predict_proba(tx)
    res = [each[0] for each in res]
    return res


if __name__ == '__main__':



    filename = './data.txt'
    x, y = load_data(filename)

    # test the Logistic model
    # from ML.LR_class import LR_class
    # LR = LR_class()

    from ML.Perceptron_class import Perceptron_class
    model = Perceptron_class()

    model.fit(x, y)
    y_pres = model.predict(x)
    print(y_pres)
    accuracy = accuracy_calculation(y,y_pres)
    print(accuracy)
    official_y_pres = LR_regression(x, y, x)
    accuracy_official = accuracy_calculation(y, official_y_pres)
    print(accuracy_official)


