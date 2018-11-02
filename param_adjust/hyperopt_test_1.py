# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 11/1/18
# file: hyperopt_test_1.py
# description:

from hyperopt import fmin, tpe, hp

best = fmin(
    fn=lambda x: x, #接受一个函数最小化
    space=hp.uniform('x', 0, 1), # 制定参数的搜索空间
    algo=tpe.suggest, #制定搜索算法
    max_evals=100) #最大评估次数
print(best)
