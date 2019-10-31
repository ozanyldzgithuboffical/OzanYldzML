# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 09:59:34 2019

@author: ozanyildiz
"""
#imports
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import array 
#reading csv
data=pd.read_csv('customers.csv')
#get age and salary data frames as arrays
dataFrame=data.iloc[:,2:4].values
#define the split into 3 folds
kfolds = KFold(n_splits=3)
 # returns the number of splitting iterations
kfolds.get_n_splits(dataFrame)
print(kfolds) 
#Create an array from 1 to 1000
Y = np.arange(1,1001)
#result
for train_index, test_index in kfolds.split(dataFrame):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = dataFrame[train_index], dataFrame[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    #initialize array with signed integer
    ('TRAIN:', array.array('i',[69900, 6325]), 'TEST:', array.array('i',[0, 1]))
