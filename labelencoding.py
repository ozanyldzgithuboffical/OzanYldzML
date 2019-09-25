# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:33:12 2019

@author: Ozan YILDIZ

About the Code:
   Converting nominal values to meaningful numerical values for data model
   In csv file there are two nominal fields:country,gender respectively
"""
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Used to convert nominal data to numerical data for machine learning model
from sklearn.preprocessing import LabelEncoder

#reading csv
data=pd.read_csv('data.csv')
#print(data)

#extracting length data frame of age & gender
#dataFrame1=data[['age','gender']]


#label encoding object is instantiated
labelEncoding=LabelEncoder()

#first we get country data values
country=data.iloc[:,0:1].values
#print(country)
country[:,0]=labelEncoding.fit_transform(country[:,0])
print(country)