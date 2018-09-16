# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 18-9-15
# file: kaggle_SVM.py
# description:

import pandas as pd
from sklearn.svm import SVC


if __name__ == '__main__':

    train_data = pd.read_csv('./train.csv')[1:]
    test_data = pd.read_csv('./train.csv')[1:]


    f = open('./data.txt','w')
    f.write(str(train_data.head()))
    f.write('zero numbers: {}'.format(train_data.isnull().sum()))
    f.write('train data shape:{}'.format(train_data.shape))
    f.write('extreme poverty: {}'.format(train_data[train_data.Target == 1].shape))
    f.write('moderate poverty : {}'.format(train_data[train_data.Target == 2].shape))
    f.write('vulnerable households {}: '.format(train_data[train_data.Target == 3].shape))
    f.write('non vulnerable households: {}'.format(train_data[train_data.Target == 4].shape))
    f.close()

    y = train_data[['Target']]
    x = train_data.iloc[:,:-1]

    missmap(test_na, col=c("black", "grey"), legend=FALSE, main='Missing Map')

    train_data.drop(['v2a1','v18q1','rez_esc',],axis=1)




    # model = SVC().fit(x,y)
    # yhat = model.predict(test_data)




