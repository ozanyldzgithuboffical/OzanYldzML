# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:47:31 2019
@author: Ozan YILDIZ
Random forest by GINI index(CART) and entropy with Information Gain (ID3)
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# data is  loaded
data = pd.read_csv('salaries.csv')

#slicing dataframe parts education_level and salary
x = data.iloc[:,1:2]
y = data.iloc[:,2:]

#Data Frames converted to array format
X = x.values
Y = y.values

#It provides the tune between smoothing hyperplane and correctly classified datapoints
criterions = ['gini', 'entropy']
predict_colors=['blue','yellow']
iterator=0
for criterionItem in criterions:
    rndforest=RandomForestClassifier(n_estimators=10,criterion=criterionItem)
    rndforest.fit(X,Y)
    plt.scatter(X,Y,color='red')
    plt.plot(X,rndforest.predict(X),color=predict_colors[iterator])
    iterator=iterator+1
    