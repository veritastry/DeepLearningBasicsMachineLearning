#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:51211 
@file: SkLearnSimple.py 
@time: 2017/07/11 
"""
from sklearn import svm

X = [[2, 0], [1, 1], [2., 3]]
y = [0, 0, 1]
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

print(clf)

# get support vectors
print(clf.support_vectors_)

# get indices of support vectors
print(clf.support_)

# get number of support vectors for each class
print(clf.n_support_)

print(clf.predict([[2, .0]]))
