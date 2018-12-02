
# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 18-11-30
# file: RF
# description:

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 导入数据
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']
print('class labels:', np.unique(df_wine['Class label']))

# 分割训练集合测试集
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# 特征值缩放-标准化，决策树模型不依赖特征缩放
# stdsc=StandardScaler()
# X_train_std=stdsc.fit_transform(X_train)
# X_test_std=stdsc.fit_transform(X_test)

# 随机森林评估特征重要性
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, n_jobs=-1, random_state=0)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    # 给予10000颗决策树平均不纯度衰减的计算来评估特征重要性
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))
# 可视化特征重要性-依据平均不纯度衰减
plt.title('Feature Importance-RandomForest')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# 在这个基础上，随机森林海可以通过阈值压缩数据集
X_selected = forest.transform(X_train, threshold=0.15)  # 大于0.15只有三个特征
print(X_selected.shape)