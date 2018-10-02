# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 10/1/18
# file: CART_classification.py
# description: CART classification and regression decision tree
# function of CART:
# 1. handle with discrete data and continuous data
# 2. handle with missing value
# 3. offer pruning
# 4. is binary tree


# note that:
# 1. solution to regression: minimize the MSE error of samples to continuous values in samples
# 2. solution to missing value: discard the features contraining missing values)
# 3. solution to pruning: pre-pruning (the official C4.5 apply the poss-pruning method)
# post-pruning reference link : https://www.cnblogs.com/allenren/p/8662504.html



import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import copy

class CART_classification():
    def __init__(self,max_deepth=10,min_leaf_sample=10):
        self.dict = None
        self.max_deepth = max_deepth
        self.min_leaf_sample = min_leaf_sample

    def _cal_Gini(self, x, y):
        N = np.shape(x)[0]
        labels = np.unique(y)
        gini_info = 0.0
        for each in labels:
            cnt = len(y[y==each])
            prob = cnt/float(N)
            gini_info += prob ** 2
        return 1 - gini_info

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

    def _major_voting(self, x, y):
        labels = np.unique(y)
        labels_cnt = [sum(y == each) for each in labels]
        return labels[np.argmax(labels_cnt)], max(labels_cnt)

    # correct
    def _regerror(self, x):
        return np.var(x) * np.shape(x)[0]

    def _is_contain_missing(self, x):
        if type(x[0]).__name__ == 'str':
            for each in x:
                if each == '':
                    return True
            return False
        else:
            for each in x:
                if each == math.inf:
                    return True
            return False


    def _choose_best_feature(self,x,y,label):
        N,dims = np.shape(x)
        x = np.array(x)
        y = np.array(y)

        entropy_base = self._cal_Gini(x, y)
        feature_best = {'feature_name':'','gain':0.0,'values':[],'feature_inx':0}
        for dim in range(dims):

            # missing value
            if self._is_contain_missing(x[:, dim]):
                continue

            feature_values = x[:,dim]
            feature_unique = np.unique(feature_values)

            if type(feature_unique[0]).__name__ != 'str':  # continuous type
                split_points = sorted(feature_unique)
                for inx, each in enumerate(split_points):
                    # split data into two parts
                    x_1_feature = [x_i[dim] if x_i[dim] <= each else 0 for x_i in x]
                    x_2_feature = [x_i[dim] if x_i[dim] > each else 0 for x_i in x]
                    reg_error = self._regerror(x_1_feature) + self._regerror(x_2_feature)
                    if reg_error < feature_best['gain'] or feature_best['gain'] == 0.0 :
                        feature_best = {'feature_name': label[dim], 'gain': reg_error, 'values': each,
                                        'feature_inx': dim}

            else: #discrete type
                entropy_sub = 0.0
                for inx,each in enumerate(feature_unique):
                    x_sub,y_sub = self._split_discrete_data(x, y, inx, each)
                    prob = len(x_sub)/float(N)
                    entropy_sub += prob * self._cal_Gini(x_sub, y_sub)
                gain = entropy_base - entropy_sub
                if gain > feature_best['gain']:
                    feature_best = {'feature_name': label[dim], 'gain': gain, 'values': feature_unique,'feature_inx':dim}
        return feature_best


    def _build_tree(self,x,y,label,deepth):
        x = np.array(x)
        y = np.array(y)
        N, dims = np.shape(x)

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
        if feature_best['feature_name'] == '':
            print(feature_best)
            print(x)
            print(y)

        Node = {'feature_name': feature_best['feature_name']}

        if type(x[0,feature_best['feature_inx']]).__name__ != 'str':# continuous type
            x_1, y_1 = self._split_continuous_data(x, y, feature_best['feature_inx'], feature_best['values'], '<=')
            x_2, y_2 = self._split_continuous_data(x, y, feature_best['feature_inx'], feature_best['values'], '>')
            label_sub = copy.deepcopy(label)
            label_sub.remove(feature_best['feature_name'])#remove selected feature from label set
            Node[feature_best['feature_name'] + '<=' + str(feature_best['values'])] = self._build_tree(x_1, y_1, label,deepth+1)
            Node[feature_best['feature_name'] + '>' + str(feature_best['values'])] = self._build_tree(x_2, y_2, label,deepth+1)

        else:
            if len(feature_best['values']) <= 2: # feature only contain two values
                for inx, each in enumerate(feature_best['values']):  # discrete type
                    x_sub, y_sub = self._split_discrete_data(x, y, feature_best['feature_inx'], each)
                    label_sub = copy.deepcopy(label)
                    label_sub.remove(feature_best['feature_name'])  # remove selected feature from label set
                    Node[feature_best['feature_name'] + '是'+ each] = self._build_tree(x_sub, y_sub, label,deepth+1)
            else:
                # left child tree
                x_sub, y_sub = self._split_discrete_data(x, y, feature_best['feature_inx'], feature_best['values'][0])
                label_sub = copy.deepcopy(label)
                label_sub.remove(feature_best['feature_name'])  # remove selected feature from label set
                Node[feature_best['feature_name'] + '是'+ feature_best['values'][0]] = self._build_tree(x_sub, y_sub, label,deepth+1)

                # rigth child tree
                x_sub_t,y_sub_t = [],[]
                for inx, each in enumerate(feature_best['values'][1:]):# discrete type
                    x_sub, y_sub = self._split_discrete_data(x, y, feature_best['feature_inx'], each)
                    x_sub_t.append(x_sub)
                    y_sub_t.append(y_sub)
                Node[feature_best['feature_name'] + '不是' + feature_best['values'][0]] = self._build_tree(x_sub, y_sub, label,deepth+1)
        return Node

    def fit(self,x,y,label):
        self.x = np.array(x)
        self.y = np.array(y)
        self.label = np.array(label)
        self.dict = self._build_tree(self.x,self.y,label,deepth=1)

    def _decode_dict(self,x,dict_copy):
        # take the root feature value
        root_feat_name = dict_copy['feature_name']
        test_value = x[list(self.label).index(root_feat_name)]

        for child in dict_copy.keys():
            if type(test_value).__name__ != str and (child.find('<=') != -1 or child.find('>') != -1):#test feature value and child value are continuous types
                if (child.find('<=') != -1 and test_value <= np.float(child.split('<=')[1])) \
                        or (child.find('>') != -1 and test_value > np.float(child.split('>')[1])):
                    if type(dict_copy[child]).__name__ == 'dict':
                        y_pred = self._decode_dict(x, dict_copy[child])
                    else:
                        y_pred = dict_copy[child]

            else:
                if child.find(root_feat_name) >=0 : # same feature
                    if (child.find('不是') >=0 and child.split('不是')[1] != test_value) \
                        or (child.find('不是') == -1 and child.split('是')[1] == test_value):
                        if type(dict_copy[child]).__name__ == 'dict':
                            y_pred = self._decode_dict(x,dict_copy[child])
                        else:
                            y_pred = dict_copy[child]

        return y_pred


    def predict(self,x):
        return [self._decode_dict(each, copy.deepcopy(self.dict)) for each in x]


if __name__ == '__main__':

    data = pd.read_csv('xigua_data.csv').ix[1:,1:]

    x_train,x_test,y_train,y_test = train_test_split(data.ix[:,:-1],data['好瓜'])
    model = CART_classification()
    model.fit(x_train, y_train,label=list(data.columns)[:-1])
    print(model.dict)
    y_pred = model.predict(np.array(x_test))
    print(y_test)
    print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print ('accuracy %f' % accuracy)
