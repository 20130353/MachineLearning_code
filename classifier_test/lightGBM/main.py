# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 11/2/18
# file: main.py
# description:

import pandas as pd
import numpy as np

air_visit = pd.read_csv('./data/air_visit_data.csv')

# 输出的是客户到饭店的数据: id, 时间, 人数
# print(air_visit.head())

# 客户到饭店的日期索引
air_visit.index = pd.to_datetime(air_visit['visit_date'])
# print(air_visit.head())

# 按照客户id统计每个客户每天到饭店的次数
air_visit = air_visit.groupby('air_store_id').apply(lambda g: g['visitors'].resample('1d').sum()).reset_index()
# print(air_visit.head())

#  缺失值填充0
air_visit['visit_date'] = air_visit['visit_date'].dt.strftime('%Y-%m-%d')
air_visit['was_nil'] = air_visit['visitors'].isnull()
air_visit['visitors'].fillna(0, inplace=True)
# print('客户到店数据')
# print(air_visit.head())

# 日历数据
date_info = pd.read_csv('./data/date_info.csv')
# print('原始日历信息')
# print(date_info.head())

# 移动日期,观察前一天和后一天是否是节假日
date_info.rename(columns={'holiday_flg': 'is_holiday', 'calendar_date': 'visit_date'}, inplace=True)
date_info['prev_day_is_holiday'] = date_info['is_holiday'].shift().fillna(0)  # 向前移动一个ie
date_info['next_day_is_holiday'] = date_info['is_holiday'].shift(-1).fillna(0)  # 向后移动一个
# print('节假日信息')
# print(date_info.head())

# 地区数据
air_store_info = pd.read_csv('./data/air_store_info.csv')
# print('地区数据')
# print(air_store_info.head())

#  测试数据集
submission = pd.read_csv('./data/sample_submission.csv')
submission['air_store_id'] = submission['id'].str.slice(0, 20)
submission['visit_date'] = submission['id'].str.slice(21)
submission['is_test'] = True
submission['visitors'] = np.nan
submission['test_number'] = range(len(submission))
# print('测试数据集')
# print(submission.head())

# 所有信息汇总
data = pd.concat((air_visit, submission.drop('id', axis='columns')))
# print('客户到店的所有信息汇总:')
# print(data.head())

data['is_test'].fillna(False, inplace=True)
data = pd.merge(left=data, right=date_info, on='visit_date', how='left')
data = pd.merge(left=data, right=air_store_info, on='air_store_id', how='left')
data['visitors'] = data['visitors'].astype(float)
# print(data.head())


import glob
weather_dfs = []
for path in glob.glob('./1-1-16_5-31-17_Weather/*.csv'):
    weather_df = pd.read_csv(path)
    weather_df['station_id'] = path.split('\\')[-1].rstrip('.csv')
    weather_dfs.append(weather_df)

weather = pd.concat(weather_dfs, axis='rows')
weather.rename(columns={'calendar_date': 'visit_date'}, inplace=True)
# print('天气信息')
# print(weather.head())

means = weather.groupby('visit_date')[['avg_temperature', 'precipitation']].mean().reset_index()
means.rename(columns={'avg_temperature': 'global_avg_temperature', 'precipitation': 'global_precipitation'}, inplace=True)
weather = pd.merge(left=weather, right=means, on='visit_date', how='left')
weather['avg_temperature'].fillna(weather['global_avg_temperature'], inplace=True)
weather['precipitation'].fillna(weather['global_precipitation'], inplace=True)
# print('平均气温')
# print(weather[['visit_date', 'avg_temperature', 'precipitation']].head())

def find_outliers(series):
    return (series - series.mean()) > 1.96 * series.std()


def cap_values(series):
    outliers = find_outliers(series)
    max_val = series[~outliers].max()
    series[outliers] = max_val
    return series

# 处理异常点
stores = data.groupby('air_store_id')
data['is_outlier'] = stores.apply(lambda g: find_outliers(g['visitors'])).values
data['visitors_capped'] = stores.apply(lambda g: cap_values(g['visitors'])).values
data['visitors_capped_log1p'] = np.log1p(data['visitors_capped'])

# 日期数据
data['is_weekend'] = data['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
# data['day_of_month'] = data['visit_date'].dt.day
data.head()

from scipy import optimize


def calc_shifted_ewm(series, alpha, adjust=True):
    return series.shift().ewm(alpha=alpha, adjust=adjust).mean()


def find_best_signal(series, adjust=False, eps=10e-5):
    def f(alpha):
        shifted_ewm = calc_shifted_ewm(series=series, alpha=min(max(alpha, 0), 1), adjust=adjust)
        corr = np.mean(np.power(series - shifted_ewm, 2))
        return corr

    res = optimize.differential_evolution(func=f, bounds=[(0 + eps, 1 - eps)])

    return calc_shifted_ewm(series=series, alpha=res['x'][0], adjust=adjust)

# 指数加权移动平均(Exponential Weighted Moving Average)，反应时间序列变换趋势，需要我们给定alpha值，这里我们来优化求一个最合适的。
# 时间模型,使最近的数据的权重更大,时间越远权重越小
roll = data.groupby(['air_store_id', 'day_of_week']).apply(lambda g: find_best_signal(g['visitors_capped']))
data['optimized_ewm_by_air_store_id_&_day_of_week'] = roll.sort_index(level=['air_store_id', 'visit_date']).values

roll = data.groupby(['air_store_id', 'is_weekend']).apply(lambda g: find_best_signal(g['visitors_capped']))
data['optimized_ewm_by_air_store_id_&_is_weekend'] = roll.sort_index(level=['air_store_id', 'visit_date']).values

roll = data.groupby(['air_store_id', 'day_of_week']).apply(lambda g: find_best_signal(g['visitors_capped_log1p']))
data['optimized_ewm_log1p_by_air_store_id_&_day_of_week'] = roll.sort_index(level=['air_store_id', 'visit_date']).values

roll = data.groupby(['air_store_id', 'is_weekend']).apply(lambda g: find_best_signal(g['visitors_capped_log1p']))
data['optimized_ewm_log1p_by_air_store_id_&_is_weekend'] = roll.sort_index(level=['air_store_id', 'visit_date']).values

# print(data.head())

# 尽可能多的提取时间序列信息
def extract_precedent_statistics(df, on, group_by):
    df.sort_values(group_by + ['visit_date'], inplace=True)

    groups = df.groupby(group_by, sort=False)

    stats = {
        'mean': [],
        'median': [],
        'std': [],
        'count': [],
        'max': [],
        'min': []
    }

    exp_alphas = [0.1, 0.25, 0.3, 0.5, 0.75]
    stats.update({'exp_{}_mean'.format(alpha): [] for alpha in exp_alphas})

    for _, group in groups:

        shift = group[on].shift()
        roll = shift.rolling(window=len(group), min_periods=1)

        stats['mean'].extend(roll.mean())
        stats['median'].extend(roll.median())
        stats['std'].extend(roll.std())
        stats['count'].extend(roll.count())
        stats['max'].extend(roll.max())
        stats['min'].extend(roll.min())

        for alpha in exp_alphas:
            exp = shift.ewm(alpha=alpha, adjust=False)
            stats['exp_{}_mean'.format(alpha)].extend(exp.mean())

    suffix = '_&_'.join(group_by)

    for stat_name, values in stats.items():
        df['{}_{}_by_{}'.format(on, stat_name, suffix)] = values


extract_precedent_statistics(
    df=data,
    on='visitors_capped',
    group_by=['air_store_id', 'day_of_week']
)

extract_precedent_statistics(
    df=data,
    on='visitors_capped',
    group_by=['air_store_id', 'is_weekend']
)

extract_precedent_statistics(
    df=data,
    on='visitors_capped',
    group_by=['air_store_id']
)

extract_precedent_statistics(
    df=data,
    on='visitors_capped_log1p',
    group_by=['air_store_id', 'day_of_week']
)

extract_precedent_statistics(
    df=data,
    on='visitors_capped_log1p',
    group_by=['air_store_id', 'is_weekend']
)

extract_precedent_statistics(
    df=data,
    on='visitors_capped_log1p',
    group_by=['air_store_id']
)

data.sort_values(['air_store_id', 'visit_date']).head()

# 划分数据集
data['visitors_log1p'] = np.log1p(data['visitors'])
train = data[(data['is_test'] == False) & (data['is_outlier'] == False) & (data['was_nil'] == False)]
test = data[data['is_test']].sort_values('test_number')

to_drop = ['air_store_id', 'is_test', 'test_number', 'visit_date', 'was_nil',
           'is_outlier', 'visitors_capped', 'visitors',
           'air_area_name', 'latitude', 'longitude', 'visitors_capped_log1p']
train = train.drop(to_drop, axis='columns')
train = train.dropna()
test = test.drop(to_drop, axis='columns')

X_train = train.drop('visitors_log1p', axis='columns')
X_test = test.drop('visitors_log1p', axis='columns')
y_train = train['visitors_log1p']
print(X_train.head())
print(y_train.head())


assert X_train.isnull().sum().sum() == 0
assert y_train.isnull().sum() == 0
assert len(X_train) == len(y_train)
assert X_test.isnull().sum().sum() == 0
assert len(X_test) == 32019

# 建立lightGBM的模型
import lightgbm as lgbm
from sklearn import metrics
from sklearn import model_selection

np.random.seed(42)

model = lgbm.LGBMRegressor(
    objective='regression',
    max_depth=5,
    num_leaves=25,
    learning_rate=0.007,
    n_estimators=1000,
    min_child_samples=80,
    subsample=0.8,
    colsample_bytree=1,
    reg_alpha=0,
    reg_lambda=0,
    random_state=np.random.randint(10e6)
)

n_splits = 6
cv = model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=42)

val_scores = [0] * n_splits

sub = submission['id'].to_frame()
sub['visitors'] = 0

feature_importances = pd.DataFrame(index=X_train.columns)

for i, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_fit = X_train.iloc[fit_idx]
    y_fit = y_train.iloc[fit_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_train.iloc[val_idx]

    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_fit, y_fit), (X_val, y_val)],
        eval_names=('fit', 'val'),
        eval_metric='l2',
        early_stopping_rounds=200,
        feature_name=X_fit.columns.tolist(),
        verbose=False
    )

    val_scores[i] = np.sqrt(model.best_score_['val']['l2'])
    sub['visitors'] += model.predict(X_test, num_iteration=model.best_iteration_)
    feature_importances[i] = model.feature_importances_

    print('Fold {} RMSLE: {:.5f}'.format(i + 1, val_scores[i]))

sub['visitors'] /= n_splits
sub['visitors'] = np.expm1(sub['visitors'])

val_mean = np.mean(val_scores)
val_std = np.std(val_scores)

print('Local RMSLE: {:.5f} (±{:.5f})'.format(val_mean, val_std))
sub.to_csv('result.csv', index=False)
