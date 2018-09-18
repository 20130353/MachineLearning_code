# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 18-9-18
# file: ID3_class.py
# description:

# -*- coding: utf-8 -*-

from numpy import *
import pandas as pd
from math import log

# calculate the xiangnong entropy
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}

    # create dict for label
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0

    # log2
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 对离散变量划分数据集，取出该特征取值为value的所有样本
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 连续变量划分数据集，direction规定划分的方向，
# 决定是划分出小于value的数据样本还是大于value的数据样本集
def splitContinuousDataSet(dataSet, axis, value, direction):
    retDataSet = []
    for featVec in dataSet:
        if direction == 0:
            if featVec[axis] > value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
        else:
            if featVec[axis] <= value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet


# select one feature to split data
def chooseBestFeatureToSplit(dataSet, labels):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    bestSplitDict = {}
    for i in range(numFeatures):

        featList = [example[i] for example in dataSet]
        # continuous data
        if type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int':
            # 产生n-1个候选划分点
            sortfeatList = sorted(featList)
            splitList = []
            for j in range(len(sortfeatList) - 1):
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2.0)

            bestSplitEntropy = 10000
            slen = len(splitList)

            # calculate the gain entropy
            for j in range(slen):
                value = splitList[j]
                newEntropy = 0.0
                subDataSet0 = splitContinuousDataSet(dataSet, i, value, 0)
                subDataSet1 = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = len(subDataSet0) / float(len(dataSet))
                newEntropy += prob0 * calcShannonEnt(subDataSet0)
                prob1 = len(subDataSet1) / float(len(dataSet))
                newEntropy += prob1 * calcShannonEnt(subDataSet1)
                if newEntropy < bestSplitEntropy:
                    bestSplitEntropy = newEntropy
                    bestSplit = j
            # save bast feature in the dict
            bestSplitDict[labels[i]] = splitList[bestSplit]
            infoGain = baseEntropy - bestSplitEntropy

        # discrete data
        else:
            uniqueVals = set(featList)
            newEntropy = 0.0

            # calculate the gain entrpoy
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    # if best feature is continuous data, split data into two parts(greater and smaller)
    if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':
        bestSplitValue = bestSplitDict[labels[bestFeature]]
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
        for i in range(shape(dataSet)[0]):
            if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0
    return bestFeature


# decide the label of data according to the number of samples
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    return max(classCount)


# create tree
def createTree(dataSet, labels, data_full, labels_full):
    classList = [example[-1] for example in dataSet]

    # one class
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # one dim
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # choose best feature
    bestFeat = chooseBestFeatureToSplit(dataSet, labels)

    # split data
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    if type(dataSet[0][bestFeat]).__name__ == 'str':
        currentlabel = labels_full.index(labels[bestFeat])
        featValuesFull = [example[currentlabel] for example in data_full]
        uniqueValsFull = set(featValuesFull)
    del (labels[bestFeat])


    # construct the subtree recursively。
    for value in uniqueVals:
        subLabels = labels[:]
        if type(dataSet[0][bestFeat]).__name__ == 'str':
            uniqueValsFull.remove(value)
        myTree[bestFeatLabel][value] = createTree(splitDataSet \
                                                      (dataSet, bestFeat, value), subLabels, data_full, labels_full)
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        for value in uniqueValsFull:
            myTree[bestFeatLabel][value] = majorityCnt(classList)
    return myTree

def classify(tree,label,testVec):
    # take the root feature value
    firstFeat = list(tree.keys())[0]
    secondDict = tree[firstFeat]

    # take the second feature value
    # if data is continuous
    if firstFeat.find('<=') != -1:
        firstFeatValue = float(firstFeat.split('<=')[1])
        firstFeat = firstFeat.split('<=')[0]
        if testVec[label.index(firstFeat)] <= firstFeatValue:
            test_value = 0
        else:
            test_value = 1
    # data is discrete
    else:
        test_value = testVec[label.index(firstFeat)]

    for key in secondDict.keys():
        if test_value == key:
            # if type is dict, it is still a subtree.
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key],label,testVec)
            else: # it is class label
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    df = pd.read_csv('xigua_data3.0.csv')
    data = df.values[:, 1:].tolist()
    data_full = data[:]
    labels = df.columns.values[1:-1].tolist()
    labels_full = labels[:]
    myTree = createTree(data, labels, data_full, labels_full)
    print(myTree)
    predictd_label = classify(myTree,labels_full,data_full[0])
    print(predictd_label)