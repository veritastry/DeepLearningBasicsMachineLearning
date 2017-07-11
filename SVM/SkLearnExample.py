#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:51211 
@file: SkLearnExample.py 
@time: 2017/07/11 
"""

# 线性可区分(linear separable)

print(__doc__)

import numpy as np
import pylab as pl
from sklearn import svm

# we create  40 separable points
np.random.seed(1)  # 使每次随机都是固定的
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]  # 正态分布，形成上下两侧点集
Y = [0] * 20 + [1] * 20  # 设定Label

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

print("w:", w)
print("a:", a)
print("xx:", xx)
print("yy:", yy)
print("support_vectors_:", clf.support_vectors_)
print("clf.coef_:", clf.coef_)

# switching to the generic n-dimensional parametrization of the hyperplane to the  2D-specific equation of a line
# y = a.x + b: the generic w_0 x + w_1 y + w_3 = 0 can be rewritten y = - (w_0 / w_1) x - (w_3 / w_1)

# plot the line, the points, and the nearest vectors to the plane
pl.plot(xx, yy, 'k-')  # 实线
pl.plot(xx, yy_down, 'k--')  # 虚线
pl.plot(xx, yy_up, 'k--')  # 虚线

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='black')  # 支持向量的点着重标出
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)  # 画出上下的点集

pl.axis('tight')
pl.show()
