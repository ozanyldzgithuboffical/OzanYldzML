# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:47:31 2019
@author: Ozan YILDIZ
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
# data is  loaded
data = pd.read_csv('salaries.csv')

#slicing dataframe parts education_level and salary
x = data.iloc[:,1:2]
y = data.iloc[:,2:]

#Data Frames converted to array format
X = x.values
Y = y.values

#criterion:gini (CART) or entropy (ID3) 
#max_leaf_nodes:Determines max number of the leaf nodes
#min_samples_leaf:Determines min number of the leaf nodes  to be left
#max_depth:determines max level of the depth of the tree
dtree=DecisionTreeClassifier(criterion="gini",splitter="random",max_leaf_nodes=10,min_samples_leaf=5,max_depth=5)
dtree.fit(X,Y)

#plotting
#draw the space between x and y
plt.scatter(X,Y,color='red')
plt.plot(X,dtree.predict(X),color='blue')

#Predicts
print(dtree.predict(15))
print(dtree.predict(5.2))