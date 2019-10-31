

# Machine Learning Repo

- **Machine Learning (ML)** is defined as the use algorithms and computational statistics to learn from data without being explicitly programmed. It is a subsection of the artificial intelligence domain within computer science.

## Logistic Regression
- Logistic regression is the most famous machine learning algorithm after linear regression. In a lot of ways, linear regression and logistic regression are similar. But, the biggest difference lies in what they are used for.
- Linear regression algorithms are used to predict/forecast values but logistic regression is used for classification tasks.

## Sigmoid Function (Logistic Function)
- Logistic regression algorithm also uses a linear equation with independent predictors to predict a value. The predicted value can be anywhere between negative infinity to positive infinity. We need the output of the algorithm to be class variable, i.e 0-no, 1-yes. Therefore, we are squashing the output of the linear equation into a range of [0,1]. To squash the predicted value between 0 and 1, we use the sigmoid function.

## Classification of Logistic Regression
1. **binomial:** target variable can have only 2 possible types: “0” or “1” which may represent “win” vs “loss”, “pass” vs “fail”, “dead” vs “alive”, etc.
2. **multinomial:** target variable can have 3 or more possible types which are not ordered(i.e. types have no quantitative significance) like “disease A” vs “disease B” vs “disease C”.
3. **ordinal:** it deals with target variables with ordered categories. For example, a test score can be categorized as:“very poor”, “poor”, “good”, “very good”. Here, each category can be given a score like 0, 1, 2, 3.


## Implementation Phase
- We first split our test and train data.Then we fit the model over X to Y.
- Than,we predict over our test data set and compare with the actual y (class) values 

- **Example Code**
```python
log_reg=LogisticRegression(random_state=0)
log_reg.fit(X_train,y_train)
yPredict=log_reg.predict(X_test)
print (yPredict)
print (y_test)
```

## Data Frame Concatanation
- Data frames are the any part of the data set.We sometimes need to make some preprocessing on the specific features.
- After the process,the dataframes should be concataned to be used as a model in learning process.
- To concatanate the data frames Pandas library used.
- **Example Code:**
```python
    #Create Data Frame for country
    countryDataFrame=pd.DataFrame(data=country,index=range(22),columns=['tr','fr','us'])
    #print(countryDataFrame)

    #Create numerical data frame
    numericalDataFrame=pd.DataFrame(data=numericalData,index=range(22),columns=['length','weight','age'])
    #print(numericalDataFrame)

    #concat two data frame I do not add the gender since it is the descriptor field of model
    ultimateDataFrame=pd.concat([countryDataFrame,numericalDataFrame],axis=1)
    print(ultimateDataFrame)
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
