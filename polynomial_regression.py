# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:47:31 2019

@author: Ozan YILDIZ
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 04:18:20 2018

@author: sadievrenseker
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# data is  loaded
data = pd.read_csv('salary.csv')

#slicing dataframe parts education_level and salary
x = data.iloc[:,1:2]
y = data.iloc[:,2:]

#Data Frames converted to array format
X = x.values
Y = y.values


#linear regression transformation
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#polynomial regression
#a polynom from second degree
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#a polynom from forth degree
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

# Plotting
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()


#Predicts
print(lin_reg.predict(15))
print(lin_reg.predict(5.2))

print(lin_reg2.predict(poly_reg.fit_transform(15)))
print(lin_reg2.predict(poly_reg.fit_transform(5.2)))









    


