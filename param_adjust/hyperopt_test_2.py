# -*- coding: utf-8 -*-
# Author: sunmengxin
# time: 11/1/18
# file: hyperopt_test_2.py
# description:
'''

这有一个更复杂的目标函数：lambda x: (x-1)**2。
这次我们试图最小化一个二次方程y(x)=(x-1)**2。

'''

# 测试复杂函数
# from hyperopt import fmin, tpe, hp
#
# best = fmin(
#     fn=lambda x: (x - 1) ** 2, #优化函数,如果想要最小值,直接取负数就可以了
#     space=hp.uniform('x', -2, 2), # 搜索空间: choice:列表,制定具体数值；normal:均值和方差；uuniform:范围的上下限
#     algo=tpe.suggest,
#     max_evals=100)
# print(best)


# 测试不同的参数搜索空间
# import hyperopt.pyll.stochastic
# space = {
#     'x': hp.uniform('x', 0, 1),
#     'y': hp.normal('y', 0, 1),
#     'name': hp.choice('name', ['alice', 'bob']),
# }
# print(hyperopt.pyll.stochastic.sample(space))

# 测试hyperopt函数内部的每个时间步的信息
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
fspace = {
    'x': hp.uniform('x', -5, 5)
}

def f(params):
    x = params['x']
    val = x**2
    return {'loss': val, 'status': STATUS_OK}

trials = Trials()
best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=50, trials=trials)
print ('best:', best)
print('trials:')
for trial in trials.trials[:2]:
    print (trial)

# 可视化具体过程:值vs时间
import matplotlib.pyplot as plt
f, ax = plt.subplots(1)
xs = [t['tid'] for t in trials.trials]
ys = [t['misc']['vals']['x'] for t in trials.trials]
ax.set_xlim(xs[0]-10, xs[-1]+10)
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
ax.set_title('$x$ $vs$ $t$ ', fontsize=18)
ax.set_xlabel('$t$', fontsize=16)
ax.set_ylabel('$x$', fontsize=16)


# 可视化具体过程:损失vs值
import matplotlib.pyplot as plt
f, ax = plt.subplots(1)
xs = [t['misc']['vals']['x'] for t in trials.trials]
ys = [t['result']['loss'] for t in trials.trials]
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
ax.set_title('$val$ $vs$ $x$ ', fontsize=18)
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$val$', fontsize=16)
