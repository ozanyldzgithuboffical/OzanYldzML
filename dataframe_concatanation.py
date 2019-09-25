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
   -Then imputed and encoded frames are concataned to obtain a whole data set object
"""
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Used to convert nominal data to numerical data for machine learning model
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#Used to impute the missing data values
from sklearn.preprocessing import Imputer

#reading csv
data=pd.read_csv('data.csv')
#read missing data csv
missingData=pd.read_csv('missingdata.csv')

#instantiation of imputer
imputerObj=Imputer(missing_values='NaN',strategy='mean',axis=0)

#get numerical ones
numericalData=missingData.iloc[:,1:4].values

#compute numerical data according to the strategy 
imputerObj=imputerObj.fit(digitalData[:,0:4])
#transform and write to data 
numericalData[:,0:4]=imputerObj.transform(numericalData[:,0:4])


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

#Create Data Frame for country
countryDataFrame=pd.DataFrame(data=country,index=range(22),columns=['tr','fr','us'])
#print(countryDataFrame)

#Create numerical data frame
numericalDataFrame=pd.DataFrame(data=numericalData,index=range(22),columns=['length','weight','age'])
#print(numericalDataFrame)

#concat two data frame I do not add the gender since it is the descriptor field of model
ultimateDataFrame=pd.concat([countryDataFrame,numericalDataFrame],axis=1)
print(ultimateDataFrame)

