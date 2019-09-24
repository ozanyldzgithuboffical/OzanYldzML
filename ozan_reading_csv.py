# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
Author Ozan YILDIZ
"""
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#reading csv
data=pd.read_csv('data.csv')
print(data)

#extracting length data frame of age & gender
dataFrame1=data[['age','gender']]

#primting data frame
print(dataFrame1)