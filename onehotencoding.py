# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:33:12 2019

@author: Ozan YILDIZ

About the Code:
   Converting nominal values to meaningful numerical values for data model
   In csv file there are two nominal fields:country,gender respectively
   I will apply one hot encoding since country field is a categorical field
   and polynominal field.I do not want the field should be calculated or
   sorted since it is a country.
   -I first need to convert nominal value to numerical value using LabelEncoder
   -Then according to the 1 the others will be 0 to recognize the country
"""
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Used to convert nominal data to numerical data for machine learning model
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

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

#instance of OneHotEncoding
oneHotEncoding=OneHotEncoder()

country=oneHotEncoding.fit_transform(country).toarray()
print(country)