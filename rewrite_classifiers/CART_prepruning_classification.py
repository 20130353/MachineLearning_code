# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 10/1/18
# file: CART_classification.py
# description: CART classification decision tree, it can solve continuous features,missing value
# note that: CART is binary decision tree
# missing value : majority class label is used to make up the missing value
# the file implemented the pre-pruning

import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import copy

class CART_classification():
    def __init__(self):
        self.dict = None
        self.max_deepth = math.inf
        self.min_data_in_leaf = 10
        self.leaves = 127

    def _cal_Gini(self, x, y):
        x = np.array(x)
        y = np.array(y)
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

    def _major_voting(self,x,y):
        labels = np.unique(y)
        labels_cnt = [len(y==each) for each in labels]
        return np.argmax(labels_cnt),max(labels_cnt)

    def _choose_best_feature(self,x,y,label):
        N,dims = x.shape
        entropy_base = self._cal_Gini(x, y)
        feature_best = {'feature_name':'','gain':0.0,'values':[],'feature_inx':0}
        for dim in range(dims):
            feature_values = x[:,dim]
            feature_unique = np.unique(feature_values)

            if type(feature_unique[0]).__name__ != 'str': #continuous type
                feature_unique = sorted(feature_unique)
                split_points = [(feature_unique[inx] + feature_unique[inx+1])/2 for inx in range(len(feature_unique)-1)]
                for inx,each in enumerate(split_points):
                    # split data into two parts
                    x_1,y_1 = self._split_continuous_data(x,y,dim,each,'<=')
                    x_2,y_2 = self._split_continuous_data(x,y,dim,each,'>')
                    entropy_sub = (len(x_1)/len(x)) * self._cal_Gini(x_1,y_1) + (len(x_2)/len(x)) * self._cal_Gini(x_2,y_2)
                    gain = entropy_base - entropy_sub
                    if gain > feature_best['gain']:
                        feature_best = {'feature_name': label[dim], 'gain': gain, 'values': each,'feature_inx': dim}

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


    def _build_tree(self,x,y,label):
        x = np.array(x)
        y = np.array(y)
        N, dims = x.shape

        # dermination condition
        if len(np.unique(y)) == 1:  # one class
            return y[0]

        if dims == 1:
            return self._major_voting(x, y)  # one dimension

        feature_best = self._choose_best_feature(x, y,label)
        Node = {'feature_name': feature_best['feature_name']}

        if type(x[0,feature_best['feature_inx']]).__name__ != 'str':# continuous type
            x_1, y_1 = self._split_continuous_data(x, y, feature_best['feature_inx'], feature_best['values'], '<=')
            x_2, y_2 = self._split_continuous_data(x, y, feature_best['feature_inx'], feature_best['values'], '>')
            copy.deepcopy(label).remove(feature_best['feature_name'])#remove selected feature from label set
            Node[feature_best['feature_name'] + '<=' + str(feature_best['values'])] = self._build_tree(x_1, y_1, label)
            Node[feature_best['feature_name'] + '>' + str(feature_best['values'])] = self._build_tree(x_2, y_2, label)

        else:
            if len(feature_best['values']) <= 2: # feature only contain two values
                for inx, each in enumerate(feature_best['values']):  # discrete type
                    x_sub, y_sub = self._split_discrete_data(x, y, feature_best['feature_inx'], each)
                    copy.deepcopy(label).remove(feature_best['feature_name'])  # remove selected feature from label set
                    Node[feature_best['feature_name'] + '是'+ each] = self._build_tree(x_sub, y_sub, label)
            else:
                # left child tree
                x_sub, y_sub = self._split_discrete_data(x, y, feature_best['feature_inx'], feature_best['values'][0])
                label_sub = copy.deepcopy(label)
                label_sub.remove(feature_best['feature_name'])  # remove selected feature from label set
                Node[feature_best['feature_name'] + '是'+ feature_best['values'][0]] = self._build_tree(x_sub, y_sub, label)

                # rigth child tree
                x_sub_t,y_sub_t = [],[]
                for inx, each in enumerate(feature_best['values'][1:]):# discrete type
                    x_sub, y_sub = self._split_discrete_data(x, y, feature_best['feature_inx'], each)
                    x_sub_t.append(x_sub)
                    y_sub_t.append(y_sub)
                Node[feature_best['feature_name'] + '不是' + feature_best['values'][0]] = self._build_tree(x_sub, y_sub, label)

        # pre-pruning
        self.dict = Node
        y_pred_extension = self.predict(x)
        label_inx, voting_count = self._major_voting(x, y)
        y_pred_nonextension = [1 if np.unique(self.y)[label_inx] == each else 0 for each in y]

        # if accuracy rate lower than that of non-spliting tree, return majority label
        if accuracy_score(y, y_pred_extension) > sum(y_pred_nonextension)/float(len(y)):
            return np.unique(self.y)[label_inx]
        else:
            return Node

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
            if type(test_value).__name__ != str and (child.find('<=') != -1 or child.find('>') != -1):#test feature value and child value are continuous types
                if (child.find('<=') != -1 and test_value <= child.split('<=')[1]) \
                        or (child.find('>') != -1 and test_value > child.split('<=')[1]):
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
        if y_pred == '':
            label_inx,voting_count = self._major_voting(self.x,self.y)
            label_unique = np.unique(self.y)
            y_pred = label_unique[label_inx]

        return y_pred


    def predict(self,x):
        res = []
        for each in x:
            y_pred = self._decode_dict(each,copy.deepcopy(self.dict))
            res.append(y_pred)
        return res


if __name__ == '__main__':

    data = pd.read_csv('xigua_data.csv').ix[1:,1:]

    x_train,x_test,y_train,y_test = train_test_split(data,data['好瓜'])
    model = CART_classification()
    model.fit(x_train, y_train,label=list(data.columns))
    # print(model.dict)
    y_pred = model.predict(np.array(x_test))
    # print(y_test)
    # print(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print ('accuracy %f' % accuracy)
