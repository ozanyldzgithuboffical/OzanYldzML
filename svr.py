# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:47:31 2019
@author: Ozan YILDIZ
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
# data is  loaded
data = pd.read_csv('salary.csv')

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


svr_reg=SVR(kernel='rbf')
svr_reg.fit(scaledX,scaledY)

# Plotting
plt.scatter(scaledX,scaledY,color='red')
plt.plot(scaledX,svr_reg.predict(scaledX), color = 'blue')
plt.show()


#Predicts
print(svr_reg.predict(15))
print(svr_reg.predict(5.2))