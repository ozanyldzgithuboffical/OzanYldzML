# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 17:47:31 2019
@author: Ozan YILDIZ
K-means algorithm
"""

#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import AgglomerativeClustering,KMeans
# data is  loaded
data = pd.read_csv('customers.csv')
#get age and salary data frames as arrays
dataFrame=data.iloc[:,2:4].values
#n_cluster:number of clusters k value
#init:method selected kmeans++ to avoid random centroid selection trap
kmeansobj=KMeans(n_clusters=4,init='k-means++')
kmeansobj.fit(dataFrame)
#looks for the cluster centers
#x:first features centroid value,y:second feature centroid value
print(kmeansobj.cluster_centers_)

#list that holds the returned values from kmeans wcss by every turn
results=[]
#here we change the k value
#random_state=fixed number to prevent randomly seeding
for i in range(1,10):
    kmeansobj=KMeans(n_clusters=i,init='k-means++',random_state=123)
    kmeansobj.fit(dataFrame)
    results.append(kmeansobj.inertia_)

#plot the wcss values
#according to the elbow point we can select the best k value to start
plt.plot(range(1,10),results)

acObj=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
X=dataFrame
Y_Predict=acObj.fit_predict(X)
print(Y_Predict)
plt.scatter(X[Y_Predict==0,0],X[Y_Predict==0,1],s=100,c='red')
plt.scatter(X[Y_Predict==1,0],X[Y_Predict==1,1],s=100,c='blue')
plt.scatter(X[Y_Predict==2,0],X[Y_Predict==2,1],s=100,c='green')
plt.scatter(X[Y_Predict==3,0],X[Y_Predict==3,1],s=100,c='yellow')
plt.scatter(X[Y_Predict==4,0],X[Y_Predict==4,1],s=100,c='purple')