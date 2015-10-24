# Machine Learning Prediction
searace  
Saturday, October 24, 2015  

##Summary##
The goal of the project is to predict the manner in which they did the barbell exercise. The "classe"  is the variable to be predicted based on several variables collected by accelerometers. See more details on the project description here (see the section on the [Weight Lifting Exercise Dataset](http://groupware.les.inf.puc-rio.br/har) ).

To get a accurate prediction model, first eliminate features with missing values. Next, is to remove the non numeric features while keeping the "classe" feature in its original state. The remain dataset is divide into three part, training set(named train), validation set(named validation) and test set. The ratio of the train dataset we use on training and validation is at 70:30 respectively. Random forest prediction model is used due to its accuracy and robustness with regard to outliers in training data.

With the 3 dataset, its purpose as below: 
- training for prediction model building. 
- validation data for comparison on accuracy and out-of-sample error(oose). 
- test for final model test.

Accuracy obtained is 99.09% and estimated out-of-sample error is 0.9% 


##Background##
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely *quantify how well they do it.* 

###Data### 
The training data are available here: 
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here: 
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

##Load all the required library##

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

##Data preprocessing##
###Download data###
Download the training and test dataset files and save them into the data sub directory. Note:This may be skip if the files are already in the location.

```r
trainUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
trainFile <- "./data/pml-training.csv"
testFile  <- "./data/pml-testing.csv"
if (!file.exists("./data")) {
    dir.create("./data")
}
if (!file.exists(trainFile)) {
    download.file(trainUrl, destfile=trainFile, method="libcurl")
}
if (!file.exists(testFile)) {
    download.file(testUrl, destfile=testFile, method="libcurl")
}
```

###Read the Data###

After downloading the data from the data source, we can read the two csv files into two data frames.

```r
trainRaw <- read.csv("./data/pml-training.csv")
testRaw <- read.csv("./data/pml-testing.csv")
dim(trainRaw);dim(testRaw)
```

```
## [1] 19622   160
```

```
## [1]  20 160
```
The train data set contains 19622 observations while test data set contains 20 observations. Both contains 160 features.

###Remove the missing values from train and test set###
Conduct a quick check that the classe does not contain NA.

```r
sum(is.na(trainRaw$classe)) == 0
```

```
## [1] TRUE
```

We proceed to remove the missing values from the datasets.

```r
trainData <- trainData[,colSums(is.na(trainData)) == 0] 
testData <- testData[,colSums(is.na(testData)) == 0]
```

###Remove the features which are not numeric and keeping the classe values###

```r
#keeping Classe variable (type: numeric)
classe <- trainData$classe

#deleting columns containing variables that are not numeric
trainNum <- trainData[,sapply(trainData, is.numeric) == 1]
testNum <- testData[,sapply(testData, is.numeric) == 1]

#deleting unnecessary columns containing "X", "timestamp","window" and "id" in headers 
unCol1 <- grepl("X|timestamp|window|id", names(trainNum))
train <- trainNum[,unCol1==0]
unCol2 <- grepl("X|timestamp|window|id", names(testNum))
test <- testNum[,unCol2==0]

#adding back classe variable to trainClean
train$classe <- classe

# dimensions of cleaned training and testing data set
dim(train); dim(test)
```

```
## [1] 19622    53
```

```
## [1] 20 52
```


##Data slicing##
The cleaned training dataset was partitioned, with about 70% of the observations allocated in the training sub-dataset training and the remaining in the validation sub-dataset validation.


```r
set.seed(6127)
inTrain <- createDataPartition(y=train$classe, p=0.7, list=FALSE)
training <- train[inTrain,]
validation <- train[-inTrain,]

dim(training)
```

```
## [1] 13737    53
```

```r
dim(validation)
```

```
## [1] 5885   53
```

Summary table of the final datasets for prediction model building.

Dataset  | No of Observation  | No of Feature 
-------  | -----------------  | ------------- 
training |13737 | 53 
validation |5885 | 53
test |20  | 52

##Modeling##
In using random trees, we conduct a 5-fold cross validation. 


```r
control <- trainControl(method="cv", 5)
modelFit <- train(classe ~ ., data=training, method="rf", trControl=control, ntree=250)
modelFit
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 10990, 10989, 10990, 10990, 10989 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9903181  0.9877527  0.001423459  0.001800598
##   27    0.9903182  0.9877526  0.002422138  0.003064393
##   52    0.9827475  0.9781747  0.002754011  0.003485109
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

We then conduct a prediction on the validation dataset using the model created.

```r
pred <- predict(modelFit, validation)
```

###Comparison of the prediction and the validation sub-dataset, with accuracy and OOSE values###

```r
confusionMatrix(pred, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672   11    0    0    0
##          B    1 1125    9    0    1
##          C    1    2 1016   16    3
##          D    0    1    1  947    6
##          E    0    0    0    1 1072
## 
## Overall Statistics
##                                           
##                Accuracy : 0.991           
##                  95% CI : (0.9882, 0.9932)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9886          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9877   0.9903   0.9824   0.9908
## Specificity            0.9974   0.9977   0.9955   0.9984   0.9998
## Pos Pred Value         0.9935   0.9903   0.9788   0.9916   0.9991
## Neg Pred Value         0.9995   0.9971   0.9979   0.9966   0.9979
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1912   0.1726   0.1609   0.1822
## Detection Prevalence   0.2860   0.1930   0.1764   0.1623   0.1823
## Balanced Accuracy      0.9981   0.9927   0.9929   0.9904   0.9953
```


```r
#accuracy of prediction model
accuracy <- postResample(pred, validation$classe)
accuracy
```

```
##  Accuracy     Kappa 
## 0.9909941 0.9886064
```


```r
#estimated out-of-sample error
oose <- 1 - accuracy[[1]]
oose 
```

```
## [1] 0.009005947
```

##Application of the model on the test data##

```r
results <- predict(modelFit, test)
results
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
