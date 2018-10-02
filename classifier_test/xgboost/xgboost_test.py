# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 10/2/18
# file: xgboost_test.py
# description:
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    train_df = train_df.fillna(train_df.mean())
    test_df = test_df.fillna(test_df.mean())
    train_data,train_label = train_df.ix[:,:-1],train_df.ix[:,-1]

    # one hot for train data and test data
    temp_data = pd.get_dummies(pd.concat([train_data,test_df]).reset_index())
    train_data = temp_data.ix[:len(train_data)-1,:]
    test_data = temp_data.ix[len(train_data):,:]

    # print('train columns', train_data.columns,'\nlen ', len(train_data.columns))
    # print('test columns', test_data.columns,'\nlen ',len(test_data.columns))
    # print('train data len',len(train_data))
    # print('train label len',len(train_label))

    # split train data into train data and test data(validation data)
    train_data,val_data,train_label,val_label = train_test_split(train_data,train_label)

    # model = xgb.XGBRegressor(seed=1850)
    # model.fit(train_data,train_label)
    # y_pred = model.predict(val_data)
    # print(r2_score(val_label,y_pred))



    # ---------------------adjust XGB parameters ---------------------------
    from sklearn import cross_validation
    from sklearn.grid_search import GridSearchCV


    param_test = {'max_depth':list(range(3,10,2)),'min_child_weight':list(range(1,6,2))}
    gssearch = GridSearchCV(estimator=xgb.XGBRegressor(base_score=0.5,
                             colsample_bylevel=1,
                             colsample_bytree=1,
                             gamma=0,
                             learning_rate=0.1,
                             max_delta_step=0,
                             missing=None,
                             n_estimators=100,
                             nthread=1,
                             objective='reg:linear',
                             reg_alpha=0,
                             reg_lambda=1,
                             scale_pos_weight=1,
                             seed=1850,
                             silent=True,
                             subsample=1),
                             param_grid=param_test,scoring='r2',cv=5)
    gssearch.fit(train_data, train_label)
    print('gssearch.grid_scores_',gssearch.grid_scores_)
    print('gssearch.best_params_',gssearch.best_params_)
    print('gssearch.best_score_',gssearch.best_score_)
    # y_pred = gssearch.predict(val_data)
    # print(r2_score(val_label, y_pred))