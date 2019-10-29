#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Now 19 

@author: ozan
predict gender from length,weight and age 
gender is male or not male
binary logistic regression
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

#2. Veri Onisleme

#2.1. Veri Yukleme
data = pd.read_csv('data.csv')

dataX=data.iloc[:,1:4].values
dataY=data.iloc[:,4:].values

#verilerin egitim ve test icin bolunmesi
x_train, x_test,y_train,y_test = train_test_split(dataX,dataY,test_size=0.33, random_state=0)

#verilerin olceklenmesi


sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

log_reg=LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)
yPredict=log_reg.predict(X_test)
print (yPredict)
print (y_test)

#we give predicted and test
cm=confusion_matrix(y_test,yPredict)
print('Confusion Matrix:')
print(cm)








    
    

