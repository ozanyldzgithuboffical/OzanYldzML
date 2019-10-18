# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 11:33:12 2019

@author: Ozan YILDIZ

About the Code:
 I will apply backward elimination algorithm on multi-linear regression 
 This algortihm works for eliminating dummy variables to get a good modeling result 
"""
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Used to impute the missing data values
from sklearn.preprocessing import Imputer
#Used to convert nominal data to numerical data for machine learning model
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#Used  to create test amd train data
from sklearn.model_selection import train_test_split
#import Linear Regression
from sklearn.linear_model import LinearRegression
#import statmodel
import  statsmodels.api as sm


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
#print(ultimateDataFrame)

#create frane for  independent gender
genderData=data.iloc[:,-1:].values
genderData[:,0]=labelEncoding.fit_transform(genderData[:,0])

#instance of OneHotEncoding
oneHotEncoding=OneHotEncoder()

genderData=oneHotEncoding.fit_transform(genderData).toarray()
genderDataFrame=pd.DataFrame(data=genderData[:,:1],index=range(22),columns=['gender'])
latestDataFrame=pd.concat([ultimateDataFrame,genderDataFrame],axis=1)
#print(genderDataFrame)
#create test,train dependent and independent variables
dependendTrain,dependentTest,independentTrain,independentTest=train_test_split(ultimateDataFrame,genderDataFrame,test_size=0.33,random_state=0)
#create instance of LinearRegression
lreg=LinearRegression()
#train data
lreg.fit(dependendTrain,independentTrain)
#then predict the trained data with 1 of 3 test data
y_predict=lreg.predict(dependentTest)

 #take length column index starts from zero
 #this time we predict length by the other features
length=latestDataFrame.iloc[:,3:4].values
leftside_features=latestDataFrame.iloc[:,:3]
rightside_features=latestDataFrame.iloc[:,4:]
ultimatedatafeatures=pd.concat([leftside_features,rightside_features],axis=1)
#now train and predictable data is ready let's train
X_Train,X_Test,Y_Train,Y_Test=train_test_split(ultimatedatafeatures,length,test_size=0.33,random_state=0)

#create instance of LinearRegression
lreg2=LinearRegression()
#train data
lreg2.fit(X_Train,Y_Train)
#then predict the trained data with 1 of 3 test data
y2_predict=lreg2.predict(X_Test)
#add the coefficient B0
data_with_beta_coefficient=np.append(arr=np.ones((22,1)).astype(int),values=ultimatedatafeatures,axis=1)
#data set to be extracted later with some columns
dataset_extracted=ultimatedatafeatures.iloc[:,[0,1,2,3,4,5]].values
#lets measure the effect of the ind. vars on dependent var:length
effect_stats=sm.OLS(endog=length,exog=dataset_extracted).fit()
#print(effect_stats.summary())
#according to the table x5 will be eliminated since p value=0.05 selected
dataset_extracted=ultimatedatafeatures.iloc[:,[0,1,2,3,5]].values
#lets measure the effect of the ind. vars on dependent var:length
effect_stats=sm.OLS(endog=length,exog=dataset_extracted).fit()
print(effect_stats.summary())
#now we can stop machine learning modelling


