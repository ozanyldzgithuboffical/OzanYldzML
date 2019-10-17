# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:33:12 2019

@author: Ozan YILDIZ

About the Code:
  Building linear regression model on sales count by months to predict
  the sales count.

"""
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Used  to create test amd train data
from sklearn.model_selection import train_test_split
#Standardization
from sklearn.preprocessing import StandardScaler

#Import linear regression library
from sklearn.linear_model import LinearRegression

#reading csv
data=pd.read_csv('salesdata.csv')
#print(data)
#seperate months data
months=data[['month']]
#seperate sales count data
salescounts=data[['salescount']]
#print(months)
#print(salescounts)

#get sales count values
salescountvals=salescounts.iloc[:,0:1].values
#print(salescountvals)
'''
#create test,train dependent and independent variables
dependentTrain,dependentTest,independentTrain,independentTest=train_test_split(months,salescounts,test_size=0.33,random_state=0)
'''
X_Train,X_Test,Y_Train,Y_Test=train_test_split(months,salescounts,test_size=0.33,random_state=0)
'''
stdScaler=StandardScaler()
X_Train=stdScaler.fit_transform(X_Train)
X_Test=stdScaler.fit_transform(X_Test)
Y_Train=stdScaler.fit_transform(X_Train)
Y_Test=stdScaler.fit_transform(X_Test)
'''
#start linear regression
#x_train=predicted values ,y_train=real values
lr=LinearRegression()
lr.fit(X_Train,Y_Train)
predict=lr.predict(X_Test)

