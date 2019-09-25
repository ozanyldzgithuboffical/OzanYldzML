# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
Author Ozan YILDIZ
About The Code:
    This code is about the dataframe extraction using iloc and its different
    selection ways
"""
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading csv
data=pd.read_csv('data.csv')
#print(data)

#extracting length data frame of age & gender
dataFrame1=data[['age','gender']]

#primting data frame
#print(dataFrame1)

#Get only second column with all rows
#.values you only get the values
dataFrame2=data.iloc[:,1:2]
#print(dataFrame2)

#Get only from 1 to 5 row of the first column:country
dataFrame3=data.iloc[1:5,0:1]
print(dataFrame3)