# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 10/2/18
# file: AdaBoost.py
# description:
# function of AdaBoost:
# implement a base framework of AdaBoost
# the effectiveness of model relies on the base estimator (here, we use the decision tree as base estimator)

from tree import Tree
import numpy as np
from sklearn.metrics import accuracy_score

class AdaBoost():

    def __init__(self,n_estimators = 10,n_samples=1024):
        self.x = None
        self.y = None
        self.N = None
        self.dims = None

        self.n_estimators = n_estimators
        self.estimators = []
        self.n_samples = n_samples
        self.weights = None
        self.alphas = []
        self.count = 0

    def _init_extimator(self):
        index = list(np.random.choice(self.N, self.n_samples, p=self.weights))

        # randomly sampling samples(no change for dimensions)
        x_sub = np.array([self.x[inx,:] for inx in index])
        y_sub = np.array([self.y[inx] for inx in index])

        while True:
            tree = Tree(x_sub,y_sub)
            y_pred = tree.predict(self.x)
            accuracy = accuracy_score(self.y,y_pred)
            if accuracy != 0.5:
                self.estimators.append(tree)
                break
        return tree,y_pred

    def fit(self,x,y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.N,self.dims = self.x.shape
        self.weights = [1/self.N for _ in range(self.N)]
        self.alphas = []

        self.count = 0
        for _ in range(self.n_estimators):
            self.count += 1

            estimator,pred = self._init_extimator() #only find one estimator with accuracy over 0.5
            error = np.array([y_ != y for y_,y in zip(self.y,pred)])

            epsilon = sum(self.weights * error)

            alpha = 0.5 * np.log((1-epsilon) / epsilon)

            z = 2 * np.sqrt((1-epsilon) * epsilon)
            agreements = [-1 if e else 1 for e in error] # y_i*h(x_i)
            self.weights = np.array([(weight / z * np.exp(-1 * alpha * agreement))
                                     for weight,agreement in zip(self.weights,agreements)])

            self.alphas.append(alpha)

    def _weight_maroity_vote(self,h):
        # select voting result with maximum classifier weight
        weighted_vote = {}
        for label, weight in h:
            if label in weighted_vote:
                weighted_vote[label] += weight
            else:
                weighted_vote[label] = weight

        max_weight = 0
        max_vote = 0
        for vote, weight in weighted_vote.items():
            if max_weight < weight:
                max_weight = weight
                max_vote = vote
        return max_vote

    def predict(self,x_test):
        y_preds = np.array([each.predict(x_test) for each in self.estimators])
        weight_y_preds = [[(pred_i,alpha) for pred_i in pred]for alpha,pred in zip(self.alphas,y_preds)]

        H = []
        for column in range(len(x_test)):  # eacn sample
            bucket = [weight_y_preds[row][column] for row in range(len(weight_y_preds))]
            H.append(bucket)

        res = [self._weight_maroity_vote(h) for h in H]
        return res


if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(data['data'], data['target'])

    # Test boost
    model = AdaBoost(n_estimators=3, n_samples=2048)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    score = accuracy_score(y_test, predictions)
    print('accuracy_score ',score)
