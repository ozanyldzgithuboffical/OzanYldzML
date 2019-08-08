//Data scaling is one of the preprocessing  technique in ML.Sometime feature vector has random values and it is useful to scale.
import numpy as np
from sklearn import preprocessing

mydata=np.array([[2,-1.2,5,6],
                 [3,2,-5.2,5],
                 [4,1,2,7,6],
                 [5,-1.5,-1.6,2]])
dataScaler=preprocessing.MinMaxScaler(feature_range=(0,1))
dataScaled=dataScaler.fit_transform(data)
print "Min max Scaled Data:",dataScaled
