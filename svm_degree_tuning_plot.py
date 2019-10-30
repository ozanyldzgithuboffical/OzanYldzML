# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:47:31 2019
@author: Ozan YILDIZ
SVM tuning
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
# data is  loaded
data = pd.read_csv('salaries.csv')

#slicing dataframe parts education_level and salary
x = data.iloc[:,1:2]
y = data.iloc[:,2:]

#Data Frames converted to array format
X = x.values
Y = y.values

#preprocessing Standardization to eliminate error data points
sc1=StandardScaler()
scaledX=sc1.fit_transform(X)
sc2=StandardScaler()
scaledY=sc2.fit_transform(Y)

#It provides the tune between smoothing hyperplane and correctly classified datapoints
degrees = [2, 3, 4, 5, 6]
predict_colors=['blue','yellow','green','orange','purple']
iterator=0

for degreeItem in degrees:
    svm=SVR(kernel='poly',degree=degreeItem)
    svm.fit(scaledX,scaledY)
    plt.scatter(scaledX,scaledY,color='red')
    plt.plot(scaledX,svm.predict(scaledX),color=predict_colors[iterator])
    iterator=iterator+1
    