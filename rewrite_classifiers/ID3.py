# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 9/30/18
# file: ID3.py
# description: decision tree with mutual information index
# function of ID3
# 1. handle with the discrete data

import numpy as np
import math
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import copy

class ID3():
    def __init__(self):
        self.dict = None

    def _cal_mutual_infomation(self,x,y):
        x = np.array(x)
        y = np.array(y)
        N = np.shape(x)[0]

        labels = np.unique(y)
        mutual_info = 0.0
        for each in labels:
            cnt = len(y[y==each])
            prob = cnt/float(N)
            mutual_info += -prob * math.log(prob,2)
        return mutual_info

    # take all samplesf with selected discrete feature
    def _split_data(self, x, y, label, current_feature_inx, feature_value):
        selected_x = []
        selected_y = []
        for inx,each in enumerate(x):
            if each[current_feature_inx] == feature_value:
                sample = np.append(each[:current_feature_inx], each[current_feature_inx + 1:])
                selected_x.append(sample)
                selected_y.append(y[inx])
        return selected_x,selected_y

    def _major_voting(self,x,y):
        labels = np.unique(y)
        labels_cnt = [sum(y == each) for each in labels]
        return labels[np.argmax(labels_cnt)], max(labels_cnt)

    def _choose_best_feature(self,x,y,label):
        N,dims = x.shape
        entropy_base = self._cal_mutual_infomation(x,y)
        gain_best = {'feature_name':'','gain':0.0,'values':[],'feature_inx':0}
        for dim in range(dims):
            feature_values = x[:,dim]
            feature_unique = np.unique(feature_values)

            entropy_sub = 0.0
            for inx,each in enumerate(feature_unique):
                x_sub,y_sub = self._split_data(x,y,label,inx,each)
                prob = len(x_sub)/float(N)
                entropy_sub += prob * self._cal_mutual_infomation(x_sub,y_sub)
            gain = entropy_base - entropy_sub
            if gain > gain_best['gain']:
                gain_best = {'feature_name': label[dim], 'gain': gain, 'values': feature_unique,'feature_inx':dim}
        return gain_best


    def _build_tree(self,x,y,label):
        x = np.array(x)
        y = np.array(y)
        N, dims = x.shape

        # dermination condition
        if len(np.unique(y)) == 1:  # one class
            return y[0]

        if dims == 1:
            label, _ = self._major_voting(x, y)  # one dimension
            return label

        feature_best = self._choose_best_feature(x, y,label)
        Node = {'feature_name': feature_best['feature_name']}
        for inx, each in enumerate(feature_best['values']):
            x_sub, y_sub = self._split_data(x, y,label, feature_best['feature_inx'], each)
            copy.deepcopy(label).remove(feature_best['feature_name'])
            Node[each] = self._build_tree(x_sub, y_sub,label)
        return Node

    # build_tree process and fit process cannot be merged, because the build_tree is interation process and it returns tree node.
    # and fit process needs to set model dict, it return nothing
    def fit(self,x,y,label):
        self.x = np.array(x)
        self.y = np.array(y)
        self.label = np.array(label)
        self.dict = self._build_tree(x,y,label)

    def _decode_dict(self,x,dict_copy):
        # take the root feature value
        root_feat_name = dict_copy['feature_name']
        test_value = x[list(self.label).index(root_feat_name)]

        y_pred = ''
        for child in dict_copy.keys():
            if test_value == child:
                # if type is dict, it is still a subtree.
                if type(dict_copy[child]).__name__ == 'dict':
                    y_pred = self._decode_dict(x,dict_copy[child])
                else:  # it is class label
                    y_pred = dict_copy[child]

        return y_pred


    def predict(self,x):
        # return [self._decode_dict(each,copy.deepcopy(self.dict)) for each in x]

        res = []
        x = x.reset_index(drop=True)
        for inx in range(len(x)):
            y_pred = self._decode_dict(x.loc[inx],copy.deepcopy(self.dict))
            res.append(y_pred)
        return res


if __name__ == '__main__':

    data = pd.read_csv('xigua_data.csv').ix[1:,1:]

    x_train,x_test,y_train,y_test = train_test_split(data.ix[:,:-3],data['好瓜'])
    model = ID3()
    model.fit(x_train, y_train,label=list(data.columns))
    # print(model.dict)

    y_pred = model.predict(x_test)
    # print(y_test)
    # print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print ('accuracy %f' % accuracy)
