# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:33:12 2019

@author: Ozan YILDIZ

About the Code:
    There are missing values on the data.We try to fill the missing 
    data using mean statistics
"""
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer

#reading csv
data=pd.read_csv('missingdata.csv')
#print(data)

#extracting length data frame of age & gender
#dataFrame1=data[['age','gender']]


imputerObj=Imputer(missing_values='NaN',strategy='mean',axis=0)

#We get the numerical columns as a dataframe with only values
digitalData=data.iloc[:,1:4].values
#print(digitalData)

#statistics is computed during fit to prevent data prevention
imputerObj=imputerObj.fit(digitalData[:,0:4])

#computed data is transformed and ready to be used in test data
digitalData[:,0:4]=imputerObj.transform(digitalData[:,0:4])
print(digitalData)