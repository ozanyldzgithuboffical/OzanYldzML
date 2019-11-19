# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:50:21 2019

@author: ozanyildiz
"""

#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori
orderData=pd.read_csv("orderbucket.csv",header=None)
orderList=[]
for i in range(0,7501):
    orderList.append([str(orderData.values[i,j]) for j in range(0,20)])
    
    
rules=apriori(orderList,min_support=0.01,min_confidence=0.2,min_lift=3,min_length=2)
print(list(rules))
    