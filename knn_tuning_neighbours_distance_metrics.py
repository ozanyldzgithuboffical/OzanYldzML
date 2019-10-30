# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:47:31 2019
@author: Ozan YILDIZ
KNN tuning by distance calculation
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
# data is  loaded
data = pd.read_csv('salaries.csv')

#slicing dataframe parts education_level and salary
x = data.iloc[:,1:2]
y = data.iloc[:,2:]

#Data Frames converted to array format
X = x.values
Y = y.values

#It provides the tune between smoothing hyperplane and correctly classified datapoints
distance_metrics = ['euclidean', 'manhattan','minkowski']
predict_colors=['blue','yellow','green']
n_neighbours=[5,6,7]
iterator=0
for neighbourItem in n_neighbours:
    iterator=0
    for metricItem in distance_metrics:
        knn_cls=KNeighborsClassifier(n_neighbors=neighbourItem,metric=metricItem)
        knn_cls.fit(X,Y)
        plt.scatter(X,Y,color='red')
        plt.plot(X,knn_cls.predict(X),color=predict_colors[iterator])
        iterator=iterator+1
    