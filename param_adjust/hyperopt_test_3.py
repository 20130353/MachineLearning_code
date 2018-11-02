# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 11/1/18
# file: hyperopt_test_3.py
# description:

from sklearn import datasets
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# 基本操作
# iris = datasets.load_iris()
# print(iris.feature_names)  # input names
# print(iris.target_names)  # output names
# print(iris.DESCR)  # everything else
#
# # 可视化类和特征
# sns.set(style="whitegrid", palette="husl")
# iris = sns.load_dataset("iris")
# print(iris.head())
# iris = pd.melt(iris, "species", var_name="measurement")
# print(iris.head())
#
# f, ax = plt.subplots(1, figsize=(15, 10))
# sns.stripplot(x="measurement", y="value", hue="species", data=iris, jitter=True, edgecolor="white", ax=ax)

# 加载数据
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# ----------------    test1:  测试k近邻    -----------------------------
def hyperopt_train_test(params):
    clf = KNeighborsClassifier(**params)
    return cross_val_score(clf, X, y).mean()
space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1, 100))
}
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
print('best:', best)

f, ax = plt.subplots(1)#, figsize=(10,10))
xs = [t['misc']['vals']['n'] for t in trials.trials]
ys = [-t['result']['loss'] for t in trials.trials]
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5)
ax.set_title('Iris Dataset - KNN', fontsize=18)
ax.set_xlabel('n_neighbors', fontsize=12)
ax.set_ylabel('cross validation accuracy', fontsize=12)