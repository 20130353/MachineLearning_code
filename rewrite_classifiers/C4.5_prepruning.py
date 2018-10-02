# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 10/1/18
# file: C4.5_prepruning.py
# description: decision tree with mutual information ratio

# note that:
# 1. missing value : majority class label is used to make up the missing value (The way to handle with the missing value is official advice.
#    In fact, offciers adcovate to discard the features contraining missing values)
# 2. regression utilize the MSE error to evaluate the rent point
# 3. the file implemented the pre-pruning (the official C4.5 apply the poss-pruning method)


import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import copy

class C45():
    def __init__(self,max_deepth=10,min_leaf_sample=10):
        self.dict = None
        self.max_deepth = max_deepth
        self.min_leaf_sample = min_leaf_sample

    def _cal_mutual_infomation(self, x, y):
        x = np.array(x)
        y = np.array(y)
        N = np.shape(x)[0]

        labels = np.unique(y)
        mutual_info = 0.0
        for each in labels:
            cnt = len(y[y == each])
            prob = cnt / float(N)
            mutual_info += -prob * math.log(prob, 2)
        return mutual_info

    # split data according to the feature_value
    # when dir is '<=', select sample with lower feature values than feature_value
    # when dir is '>', select sample with bigger feature values than feature_value
    def _split_continuous_data(self,x,y,current_feature_inx,feature_value,dir):
        selected_x = []
        selected_y = []
        for inx, each in enumerate(x):

            try:

                if dir == '<=' and each[current_feature_inx] <= feature_value:

                    sample = np.append(each[:current_feature_inx], each[current_feature_inx + 1:])
                    selected_x.append(sample)
                    selected_y.append(y[inx])

                if dir == '>' and each[current_feature_inx] >= feature_value:
                    sample = np.append(each[:current_feature_inx], each[current_feature_inx + 1:])
                    selected_x.append(sample)
                    selected_y.append(y[inx])
            except Exception:
                print('Error:')
                print('inx,each:')
                print(inx,each)
                print('\neach[current_feature_inx]')
                print(each[current_feature_inx])
                print('\nfeature_value')
                print(feature_value)
        return selected_x, selected_y

    # take all samplesf with selected discrete feature
    def _split_discrete_data(self, x, y, current_feature_inx, feature_value):
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
        labels_cnt = [sum(y==each) for each in labels]
        return labels[np.argmax(labels_cnt)],max(labels_cnt)

    def _choose_best_feature(self,x,y,label):
        N,dims = x.shape
        entropy_base = self._cal_mutual_infomation(x, y)

        feature_best = {'feature_name': '', 'gain': 0.0, 'values': [], 'feature_inx': 0}
        for dim in range(dims):
            feature_values = x[:,dim]
            feature_unique = np.unique(feature_values)
            if type(feature_unique[0]).__name__ != 'str': #continuous type
                feature_best = {'feature_name': '', 'gain': math.inf, 'values': [], 'feature_inx': 0}
                feature_unique = sorted(feature_unique)
                split_points = [(feature_unique[inx] + feature_unique[inx+1])/2 for inx in range(len(feature_unique)-1)]
                for inx,each in enumerate(split_points):
                    reg_error = sum([(x_i[dim]-each)** 2 if x_i[dim] <= each else 0 for x_i in x]) \
                                + sum([(x_i[dim]-each)** 2 if x_i[dim] > each else 0 for x_i in x])
                    if (reg_error < feature_best['gain']) or (feature_best['gain'] == 0.0):
                        feature_best = {'feature_name': label[dim], 'gain': reg_error, 'values': each,'feature_inx': dim}

            else: #discrete type
                entropy_sub = 0.0
                for inx,each in enumerate(feature_unique):
                    x_sub,y_sub = self._split_discrete_data(x, y, dim, each)
                    prob = len(x_sub)/float(N)
                    entropy_sub += prob * self._cal_mutual_infomation(x_sub, y_sub)
                try:
                    gain = (entropy_base - entropy_sub)/entropy_sub
                except Exception:
                    gain = (entropy_base - entropy_sub)
                if gain > feature_best['gain']:
                    feature_best = {'feature_name': label[dim], 'gain': gain, 'values': feature_unique,'feature_inx':dim}
        return feature_best


    def _build_tree(self,x,y,label,deepth):
        x = np.array(x)
        y = np.array(y)
        N, dims = x.shape

        # dermination condition
        if len(np.unique(y)) == 1:  # one class
            return y[0]

        if dims == 1:
            label, _ = self._major_voting(x, y)  # one dimension
            return label

        # prepruning
        if len(y) < self.min_leaf_sample or deepth == self.max_deepth:
            label, _ = self._major_voting(x, y)  # one dimension
            return label

        feature_best = self._choose_best_feature(x, y,label)
        Node = {'feature_name': feature_best['feature_name']}
        if type(x[0,feature_best['feature_inx']]).__name__ != 'str':# continuous type
            x_1, y_1 = self._split_continuous_data(x, y, feature_best['feature_inx'], feature_best['values'], '<=')
            x_2, y_2 = self._split_continuous_data(x, y, feature_best['feature_inx'], feature_best['values'], '>')
            copy.deepcopy(label).remove(feature_best['feature_name'])#remove selected feature from label set
            Node[feature_best['feature_name'] + '<=' + str(feature_best['values'])] = self._build_tree(x_1, y_1, label, deepth+1)
            Node[feature_best['feature_name'] + '>' + str(feature_best['values'])] = self._build_tree(x_2, y_2, label, deepth+1)
        else:
            feature_best = self._choose_best_feature(x, y, label)
            Node = {'feature_name': feature_best['feature_name']}
            for inx, each in enumerate(feature_best['values']):
                x_sub, y_sub = self._split_discrete_data(x, y, feature_best['feature_inx'], each)
                copy.deepcopy(label).remove(feature_best['feature_name'])
                Node[each] = self._build_tree(x_sub, y_sub, label, deepth+1)
        return Node

    def fit(self,x,y,label):
        self.x = np.array(x)
        self.y = np.array(y)
        self.label = np.array(label)
        self.dict = self._build_tree(x,y,label,deepth=1)

    def _decode_dict(self,x,dict_copy):
        # take the root feature value
        root_feat_name = dict_copy['feature_name']
        test_value = x[list(self.label).index(root_feat_name)]

        for child in dict_copy.keys():
            if type(test_value).__name__ != str and (child.find('<=') != -1 or child.find('>') != -1):#test feature value and child value are continuous types
                if (child.find('<=') != -1 and test_value <= np.float(child.split('<=')[1])) \
                        or (child.find('>') == -1 and test_value > np.float(child.split('<=')[1])):
                    if type(dict_copy[child]).__name__ == 'dict':
                        y_pred = self._decode_dict(x, dict_copy[child])
                    else:
                        y_pred = dict_copy[child]

            else:
                if child.find(root_feat_name) >=0 : # same feature
                    if type(dict_copy[child]).__name__ == 'dict':
                        y_pred = self._decode_dict(x, dict_copy[child])
                    else:
                        y_pred = dict_copy[child]
        return y_pred


    def predict(self,x):

        return [self._decode_dict(each, copy.deepcopy(self.dict)) for each in x]


if __name__ == '__main__':

    data = pd.read_csv('xigua_data.csv').ix[1:,1:]

    x_train,x_test,y_train,y_test = train_test_split(data.ix[:,:-1],data['好瓜'])

    model = C45()
    model.fit(x_train, y_train,label=list(data.columns))
    print('dict')
    print(model.dict)
    y_pred = model.predict(np.array(x_test))
    # print('y_true')
    # print(y_test)
    # print('y_pred')
    # print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print ('accuracy %f' % accuracy)