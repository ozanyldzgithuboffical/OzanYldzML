import numpy as np
from sklearn import preprocessing

mydata=np.array([[2,-1.2,5,6],
                 [3,2,-5.2,5],
                 [4,1,2,7,6],
                 [5,-1.5,-1.6,2]])
stdData=preprocessing.scale(mydata)
print("\nMean =", stdData.mean(axis=0))
print("Std deviation =", stdData.std(axis=0))
