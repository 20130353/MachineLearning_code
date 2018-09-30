# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 18-9-19
# file: ADAboost_class.py
# description: little test

import numpy as np

def loadSimData():
    '''
    输入：无
    功能：提供一个两个特征的数据集
    输出：带有标签的数据集
    '''
    datMat = np.matrix([[1. ,2.1],[2. , 1.1],[1.3 ,1.],[1. ,1.],[2. ,1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

def stumpClassify(dataMatrix,dimen,thresholdValue,thresholdIneq):
    '''
    输入：数据矩阵，特征维数，某一特征的分类阈值，分类不等号
    功能：输出决策树桩标签
    输出：标签
    '''
    returnArray = np.ones((np.shape(dataMatrix)[0],1))
    if thresholdIneq == 'lt':
        returnArray[dataMatrix[:,dimen] <= thresholdValue] = -1
    else:
        returnArray[dataMatrix[:,dimen] > thresholdValue] = -1
    return returnArray


# 建立基分类器是通过某个维度上的最小值来确定的
def buildStump(dataArray,classLabels,D):
    '''
    输入：数据矩阵，对应的真实类别标签，特征的权值分布
    功能：在数据集上，找到加权错误率（分类错误率）最小的单层决策树，显然，该指标函数与权重向量有密切关系
    输出：最佳树桩（特征，分类特征阈值，不等号方向），最小加权错误率，该权值向量D下的分类标签估计值
    '''
    dataMatrix = np.mat(dataArray); labelMat = np.mat(classLabels).T
    m,n = np.shape(dataMatrix)
    stepNum = 10.0; bestStump = {}; bestClassEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n): #特征维度
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/stepNum
        for j in range(-1, int(stepNum)+1): #维度数值区间的步长循环
            for thresholdIneq in ['lt', 'gt']:
                thresholdValue =  rangeMin + float(j) * stepSize
                predictClass = stumpClassify(dataMatrix,i,thresholdValue,thresholdIneq) #小于阈值分类结果是-1，大于阈值分类结果是+1
                errArray =  np.mat(np.ones((m,1)))
                errArray[predictClass == labelMat] = 0
                weightError = D.T * errArray
                #print "split: dim %d, thresh: %.2f,threIneq:%s,weghtError %.3F" %(i,thresholdValue,thresholdIneq,weightError)
                if weightError < minError:
                    minError = weightError
                    bestClassEst = predictClass.copy()
                    bestStump['dimen'] = i
                    bestStump['thresholdValue'] = thresholdValue
                    bestStump['thresholdIneq'] = thresholdIneq
    return bestClassEst, minError, bestStump

def adaBoostTrainDS(dataArray,classLabels,numIt=40):
    '''
    输入：数据集，标签向量，最大迭代次数
    功能：创建adaboost加法模型
    输出：多个弱分类器的数组
    '''
    weakClass = []#定义弱分类数组，保存每个基本分类器bestStump
    m,n = np.shape(dataArray)
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        print( "i:",i)
        bestClassEst, minError, bestStump = buildStump(dataArray,classLabels,D)#step1:找到最佳的单层决策树
        print( "D.T:", D.T)
        alpha = float(0.5*np.log((1-minError)/max(minError,1e-16)))#step2: 更新alpha
        print("alpha:",alpha)
        bestStump['alpha'] = alpha
        weakClass.append(bestStump)#step3:将基本分类器添加到弱分类的数组中
        print( "classEst:",bestClassEst)
        expon = np.multiply(-1*alpha*np.mat(classLabels).T,bestClassEst)
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()#step4:更新权重，该式是让D服从概率分布
        aggClassEst += alpha*bestClassEst#steo5:更新累计类别估计值
        print( "aggClassEst:",aggClassEst.T)
        print( np.sign(aggClassEst) != np.mat(classLabels).T)
        aggError = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,np.ones((m,1)))
        print( "aggError",aggError)
        aggErrorRate = aggError.sum()/m
        print( "total error:",aggErrorRate)
        if aggErrorRate == 0.0: break
    return weakClass

def adaTestClassify(dataToClassify,weakClass):
    dataMatrix = np.mat(dataToClassify)
    m =np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(weakClass)):
        classEst = stumpClassify(dataToClassify,weakClass[i]['dimen'],weakClass[i]['thresholdValue']\
                                 ,weakClass[i]['thresholdIneq'])
        aggClassEst += weakClass[i]['alpha'] * classEst
        print(aggClassEst)

    return np.sign(aggClassEst)

if __name__  ==  '__main__':
    D =np.mat(np.ones((5,1))/5)
    dataMatrix ,classLabels= loadSimData()
    bestClassEst, minError, bestStump = buildStump(dataMatrix,classLabels,D)
    weakClass = adaBoostTrainDS(dataMatrix,classLabels,9)
    testClass = adaTestClassify(np.mat([0,0]),weakClass)
