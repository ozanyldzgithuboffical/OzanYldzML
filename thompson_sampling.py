# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:47:04 2019

@author: ozanyildiz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math
data=pd.read_csv('AdsData.csv')

#click count
N=10000
#total ads count
adscount=10
#for every adds open a bucket for rewardd
#Ri(n)
rewards=[0]*adscount
#clicks until the time t
#Ni(n)
clickcount=[0]*adscount
#selected add for every time t
selectedads=[]
#total reward
total=0
for n in range(1,N):
    #selected advetisement
    ad=0
    max_ucb=0
    for i in range(0,adscount):
        if (clickcount[i] > 0) :
            average=rewards[i]/clickcount[i]
            delta=math.sqrt(3/2*math.log(n)/clickcount[i])
            ucb=average+delta
        else:
            ucb=N*10
        if max_ucb<ucb:
            max_ucb=ucb
            ad=i
            
    #hold in a list in which advertisement selected on the row
    selectedads.append(ad)
    #update click count for selected ad
    clickcount[ad]=clickcount[ad]+1
    #get the reward for selected advertisement
    reward=data.values[n,ad]
    rewards[ad]=rewards[ad]+reward
    total=total+reward
    
#historgram of selected ads
print("Toplam Ödül")
print(total)
plt.hist(selectedads)
plt.show()