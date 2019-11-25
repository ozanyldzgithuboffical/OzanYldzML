# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:47:04 2019

@author: ozanyildiz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
data=pd.read_csv('AdsData.csv')

#random generator we need
N=10000
adscount=10
reward=0
selectedads=[]
total=0
for n in range(0,N):
    #can click one of ten ads
    ad=random.randrange(adscount)
    #hold in a list in which advertisement selected on the row
    selectedads.append(ad)
    #get the value of adv to get the reward
    reward=data.values[n,ad]
    #calculate the total reward
    total=total+reward
    
#historgram of selected ads
plt.hist(selectedads)
plt.show()