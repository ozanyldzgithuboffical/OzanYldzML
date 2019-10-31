

# Machine Learning Repo

- **Machine Learning (ML)** is defined as the use algorithms and computational statistics to learn from data without being explicitly programmed. It is a subsection of the artificial intelligence domain within computer science.

## Multiple Linear Regression
- If linear regression is just the plotting of a relationship between an independent variable (X) and a dependent variable (Y), you may be able to guess that **multivariate/multiple linear regression** is just a linear regression carried out on more than one independent variable.
- Y=a+b1∗X1+b2∗x2
- We first split our test and train data.Then we fit the model over X to Y.
- Than,we predict over our test data set and compare with the actual y (class) values 

- **Example Code**
```python
    #now train and predictable data is ready let's train
    X_Train,X_Test,Y_Train,Y_Test=train_test_split(ultimatedatafeatures,length,test_size=0.33,random_state=0)

    #create instance of LinearRegression
    lreg2=LinearRegression()
    #train data
    lreg2.fit(X_Train,Y_Train)
    #then predict the trained data with 1 of 3 test data
    y2_predict=lreg2.predict(X_Test)
```
## Dummy Variable Trap
- For transforming categorical attribute to numerical attribute, we can use label encoding procedure (label encoding assigns a unique integer to each category of data). But this procedure is not alone that much suitable, hence, One hot encoding is used in regression models following label encoding. This enables us to create new attributes according to the number of classes present in the categorical attribute i.e if there are n number of categories in categorical attribute, n new attributes will be created. These attributes created are called **Dummy Variables**. Hence, dummy variables are **“proxy”** variables for categorical data in regression models.
- These dummy variables will be created with **one hot encoding** and each attribute will have value either 0 or 1, representing presence or absence of that attribute.

## Backward Elimination Algorithm
- Multiple linear regression model implementation with automated backward elimination (with p-value and adjusted r-squared) in Python and R for showing the relationship among profit and types of expenditures and the states.
- Before we dive into Backward elimination, let’s first understand the following terms **Statistical hypotheses** and **P-Value**
- A **P-value** helps determine weather a hypothesis must be accepted or rejected.
- Once we have a basic understanding of the above, we can jump straight into Backward Elimination. Typically, it can be performed in just 4 simple steps:
1. Select a significance level, say 5% (0.05)
2. Fit a model with all features (variables)
3. Consider the feature with the highest P-Value. If its P-value is greater than significance level (P > SL), go to step 4. Else, your model is ready.
4. Eliminate this feature (variable).
5. Fit a model with the new set of features, and go to step 3.
- **Example Code**
```python
 #take length column index starts from zero
 #this time we predict length by the other features
length=latestDataFrame.iloc[:,3:4].values
leftside_features=latestDataFrame.iloc[:,:3]
rightside_features=latestDataFrame.iloc[:,4:]
ultimatedatafeatures=pd.concat([leftside_features,rightside_features],axis=1)
#now train and predictable data is ready let's train
X_Train,X_Test,Y_Train,Y_Test=train_test_split(ultimatedatafeatures,length,test_size=0.33,random_state=0)

#create instance of LinearRegression
lreg2=LinearRegression()
#train data
lreg2.fit(X_Train,Y_Train)
#then predict the trained data with 1 of 3 test data
y2_predict=lreg2.predict(X_Test)
#add the coefficient B0
data_with_beta_coefficient=np.append(arr=np.ones((22,1)).astype(int),values=ultimatedatafeatures,axis=1)
#data set to be extracted later with some columns
dataset_extracted=ultimatedatafeatures.iloc[:,[0,1,2,3,4,5]].values
#lets measure the effect of the ind. vars on dependent var:length
effect_stats=sm.OLS(endog=length,exog=dataset_extracted).fit()
#print(effect_stats.summary())
#according to the table x5 will be eliminated since p value=0.05 selected
dataset_extracted=ultimatedatafeatures.iloc[:,[0,1,2,3,5]].values
#lets measure the effect of the ind. vars on dependent var:length
effect_stats=sm.OLS(endog=length,exog=dataset_extracted).fit()
print(effect_stats.summary())
#now we can stop machine learning modelling
```
- Here we use here **Ordinary Least Squares (OLS)** to see the value is equal or smaller than **p-value**
- For more information about [OLS](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html)

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
