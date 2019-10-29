#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Now 19 

@author: ozan
predict gender from length,weight and age 
gender is male or not male
KNN classification
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

#2. Veri Onisleme

#2.1. Veri Yukleme
data = pd.read_csv('data.csv')

dataX=data.iloc[:,1:4].values
dataY=data.iloc[:,4:].values

#verilerin egitim ve test icin bolunmesi
x_train, x_test,y_train,y_test = train_test_split(dataX,dataY,test_size=0.33, random_state=0)

#verilerin olceklenmesi
svc_classificator=SVC(kernel='poly')
svc_classificator.fit(x_train,y_train)
svc_predict=svc_classificator.predict(x_test)

svc2_classificator=SVC(kernel='rbf')
svc2_classificator.fit(x_train,y_train)
svc2_predict=svc2_classificator.predict(x_test)

#we give predicted and test
cm=confusion_matrix(y_test,svc_predict)
cm2=confusion_matrix(y_test,svc2_predict)

print('Confusion Matrix Polynomial:')
print(cm)

print('Confusion Matrix RBF:')
print(cm2)






    
    

