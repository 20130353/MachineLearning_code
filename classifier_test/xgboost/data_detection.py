# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 10/2/18
# file: data_detection.py
# description:
import pandas as pd

def status(x) :
    return pd.Series([x.count(),x.min(),x.quantile(.25),x.median(),
                      x.quantile(.75),x.max(),x.mean(),x.std,x.idxmin(),x.idxmax()],
                     index=['总数','最小值','25%分位数','中位数','75%分位数','最大值','均值','标准差','最小值位置','最大值位置'])


if __name__ == '__main__':

    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')


    print('shape is ', train_df.shape)
    print('desciption is\n',train_df.describe())

    # ------------  missing value     -----------------------

    # print('missing columns are\n',list(train_df.isnull().any()))
    print('missing columns number is',sum(train_df.isnull().any()))
    print('missing ratio is ', sum(train_df.isnull().any())/(float(train_df.shape[1]-1)))

    # -----------  data dtype -------------------
    types_df = pd.DataFrame(train_df.dtypes,columns=['type'])
    print('type count are\n',types_df['type'].value_counts())


    # ------------ y value ----------------
    print('max min',(train_df['SalePrice'].max(),train_df['SalePrice'].min()))
    df = pd.DataFrame(status(train_df['SalePrice']))
    print(df)



    #################    preprocessing       ################################
    train_df.fillna(train_df.mean())
    print('desciption is\n', train_df.describe())


