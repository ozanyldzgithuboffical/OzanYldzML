# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:33:12 2019

@author: Ozan YILDIZ

About the Code:
  Measuring recall,precision,sensivity,f1 measure,confusion matrix

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

from sklearn.metrics import precision_score,recall_score,f1_score,confusion_matrix

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

#start linear regression
#x_train=predicted values ,y_train=real values
lr=LinearRegression()
lr.fit(months,salescounts)
predict=lr.predict(months)


#precision
print("Precision:")
print(precision_score(salescounts,predict))
#recall
print("Recall:")
print(recall_score(salescounts,predict))
#f1 score
print("F1 Score:")
print(f1_score(salescounts,predict))
#confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(salescounts,predict))

