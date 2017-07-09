#!usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:51211 
@file: SkLearnExample.py 
@time: 2017/07/09 
"""
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()

iris = datasets.load_iris()

print(iris)

knn.fit(iris.data, iris.target)

predictedLabel = knn.predict([[0.1, 0.2, 0.3, 0.4]])

print(predictedLabel)
