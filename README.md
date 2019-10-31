

# Machine Learning Repo

- **Machine Learning (ML)** is defined as the use algorithms and computational statistics to learn from data without being explicitly programmed. It is a subsection of the artificial intelligence domain within computer science.

## K-Nearest Neighbours Classifier (K-NN)
- The KNN algorithm assumes that similar things exist in close proximity. In other words, **similar things are near to each other**.

## Algorithm Steps
- Load the data
- Initialize K to your chosen number of neighbors
-  For each example in the data
- Calculate the distance between the query example and the current example from the data.
- Add the distance and the index of the example to an ordered collection
- Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
- Pick the first K entries from the sorted collection
- Get the labels of the selected K entries
- If regression, return the mean of the K labels
- If classification, return the mode of the K labels

## Eager Vs. Lazy Learners
- Eager learners mean when given training points will construct a generalized model before performing prediction on given new points to classify. You can think of such learners as being ready, active and eager to classify unobserved data points.

- Lazy Learning means there is no need for learning or training of the model and all of the data points used at the time of prediction. Lazy learners wait until the last minute before classifying any data point. Lazy learner stores merely the training dataset and waits until classification needs to perform. Only when it sees the test tuple does it perform generalization to classify the tuple based on its similarity to the stored training tuples. Unlike eager learning methods, lazy learners do less work in the training phase and more work in the testing phase to make a classification. Lazy learners are also known as instance-based learners because lazy learners store the training points or instances, and all learning is based on instances.

## Choosing the right value for K
- Research has shown that no optimal number of neighbors suits all kind of data sets. Each dataset has it's own requirements. In the case of a small number of neighbors, the noise will have a higher influence on the result, and a large number of neighbors make it computationally expensive. 
- Research has also shown that a small amount of neighbors are most flexible fit which will have low bias but high variance and a large number of neighbors will have a smoother decision boundary which means lower variance but higher bias.

- Generally, Data scientists choose as an odd number if the number of classes is even. You can also check by generating the model on different values of k and check their performance. You can also try Elbow method here.

## Distance Metric Types
- There are different distance calculation theorems that you can apply on K-NN classifier.
- **Metrics intended for real-valued vector spaces:**

- identifier	 class name	 args	distance function
- “euclidean”	EuclideanDistance	      sqrt(sum((x - y)^2))
- “manhattan” 	ManhattanDistance	    sum(|x - y|)
- “chebyshev”  	ChebyshevDistance	    max(|x - y|)
- “minkowski”	  MinkowskiDistance	  p	sum(|x - y|^p)^(1/p)
- “wminkowski”	WMinkowskiDistance	p, w	sum(|w * (x - y)|^p)^(1/p)
- “seuclidean”	SEuclideanDistance	V	sqrt(sum((x - y)^2 / V))
- “mahalanobis”	MahalanobisDistance	V or VI	sqrt((x - y)' V^-1 (x - y))

## K-Means ++
- The default of init is k-means++ which is supposed to yield a better results than just random initialization of centroids.

- **Example Code**
```python
knn=KNeighborsClassifier(n_neighbors=2,metric='minkowski')
knn.fit(x_train,y_train)
knn_predict=knn.predict(x_test)



#we give predicted and test
cm=confusion_matrix(y_test,knn_predict)
print('Confusion Matrix:')
print(cm)

```

## Data Frame Concatanation
- Data frames are the any part of the data set.We sometimes need to make some preprocessing on the specific features.
- After the process,the dataframes should be concataned to be used as a model in learning process.
- To concatanate the data frames Pandas library used.
- **Example Code:**
```python
#Data Frames converted to array format
X = x.values
Y = y.values

#It provides the tune between smoothing hyperplane and correctly classified datapoints
distance_metrics = ['euclidean', 'manhattan','minkowski']
predict_colors=['blue','yellow','green']
n_neighbours=[5,6,7]
iterator=0
for neighbourItem in n_neighbours:
    iterator=0
    for metricItem in distance_metrics:
        knn_cls=KNeighborsClassifier(n_neighbors=neighbourItem,metric=metricItem)
        knn_cls.fit(X,Y)
        plt.scatter(X,Y,color='red')
        plt.plot(X,knn_cls.predict(X),color=predict_colors[iterator])
        iterator=iterator+1
```

## Data Frame Extraction
- In machine learning we work on datasets.These datasets consists of features that we extract.
- In some cases we could not find any feature or an input that is about un-supervised learning topic.
- These datasets are also a data frame themselves and we can take a small part of them.For instance we can use some part of the data
for model learning as independent variables and some others for dependent variables to predicted.
- Pandas library helps us to extract data frames.

## Source of Data

- It can be any unprocessed fact, value, text, sound or picture that is not being interpreted and analyzed. 
- Data is the most important part of all Data Analytics, Machine Learning, Artificial Intelligence. Without data, we can’t train any model and all modern research and automation will go vain. 
- Big Enterprises are spending loads of money just to gather as much certain data as possible.
- Data can be sourced from anywhere such as via sensors,human actions,emotions etc.The most important thing is to understand the data and extracting knowledge with different kinds of methods.

## Data Types

- There are two major data types categorical & numerical data types respectively.
- Some machine learning algorithms do not work with nominal data types.We can need an encoding to provide an understanding for the algorithm.For instance think about the cities in your country.There are n cities and in some way we need to create an indicator for those to extract the feature.
- Categorical data types are divided into two major groups, nominal & ordinal data types.
- Ordinal data types can be sorted but not measured such as plate numbers.
- Nominal data types neither measured nor ordered.
- To convert the nominal and ordinal data types into measurable data types we can use some methods like Label Encoding,One-Hot Encoding.
- Numerical data types are divided into two major group: rational and interval data types.Values of these data types can be continuous like temperature sensor values etc.

## Missing Values
1. Sometimes our dataset can consist of missing values.In such conditions there are some methods to eliminate this factor.These are called **Imputation Methods**.Let's tell about some of them
- **Listweise Deletion**
 * Only data having missing records deleted simply.
- **Pairweise Deletion**
 * Only data entry deleted which has mising value on any feature/variable/column
- **Regression Imputation**
 * Missing value marked is copied with the older entry
- **Sthocastic Imputation**
 * A statistical distribution obtained over a data set and this distribution is used as a connection to fill the missing value record.

## Data Frames
- Data Frames are the part of data set.You can make a partition,take some entries of the data set.
- Pandas library is used to create data frames.

## Feature Scaling
- Sometimes our data set features values can be unproportional in terms of their lower and upper bound values or mean values.Plus,such machine learning algorithms have weakness for such cases,for instance **support vector machine (SVM)** . There are two known methods to tune the record values of different features to make the prediction in a healthy way.
-Every data set does not require normalization only for the data sets with which features have different range.
1.**Normalization**
-Normalization is a technique often applied as part of data preparation for machine learning. The goal of normalization is to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values.
- z=(x-min(x))/(max(x)-min(x) where x is the value of the data set record.
- Drawback of the normalization is,if there is a very big value in data set,then other datas can be reduced to 0 except the normalized data becomes 1.Because of this sometimes normalization can lead to lose the data in feature.

2. **Standardization**
Standardization looks for the deviation from the mean value without losing the value importance.

## Announcement
- Overview of Deep Learning, **Dimension Reduction** , **Model Selection** , **XGBoot** topics will be under **Deep Learning Repo** !
- **Convolutional Neural Networks (CNN)** will be under **Artificial Intelligence Repo (AI)** !
- **Computer Vision** , **Self Driving Autonomous** with Tensorflow-Keras & Computer Vision & Deep Learning Repos will be also shared !
- **Kubernates** will be also shared !

## About the Repo
- This repo is open-source and aims at giving an overview about the top-latest topics that will lead learning of the basis of intelligent systems basis .

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate. Thanks.

**OZAN YILDIZ**
-Computer Engineer at HAVELSAN Ankara/Turkey 
**Linkedin**
[Ozan YILDIZ Linkedin](https://www.linkedin.com/in/ozan-yildiz-b8137a173/)

## License
[MIT](https://choosealicense.com/licenses/mit/)
