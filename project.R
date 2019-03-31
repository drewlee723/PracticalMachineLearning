### Course Project ###

library(caret)
library(corrplot)
library(rattle)
set.seed(19331)

#Read the training data and replace empty values by NA
url_training <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv'
url_testing <- 'https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv'
# download.file(url = url_training, destfile = 'data_train.csv', method="curl")
# download.file(url = url_testing, destfile = 'data_test.csv', method="curl")
train_raw <- read.csv(file = 'data_train.csv',
                      na.strings = c('NA','#DIV/0!',''), header=TRUE)
test_raw <- read.csv(file = 'data_test.csv',
                     na.strings = c('NA','#DIV/0!',''), header=TRUE)
#Identify set column name differences
setdiff(names(train_raw), names(train_raw))
setdiff(names(test_raw), names(test_raw))
#verified that the schema of both the training and testing sets are identical 
#(excluding the final column representing the A-E class), 


### Cross Validation ###
#Create a partition with the training dataset 
inTrain  <- createDataPartition(train_raw$classe, p=0.75, list=FALSE)
training <- train_raw[inTrain, ]
testing  <- train_raw[-inTrain, ]
dim(training); dim(testing)


### EDA and cleaning data ###

#Check the dimesion
dim(training)     # created datasets have 160 variables.

#Remove non-predictive variables.
training <- training[,-1:-7]
testing <- testing[,-1:-7]

#Remove variables with Nearly Zero Variance
NZV <- nearZeroVar(training)
training <- training[, -NZV]
NZV <- nearZeroVar(testing)
testing <- testing[, -NZV]

#Remove variables with missing data >90%
round(sum(is.na(training))/prod(dim(training))*100)
count_nas <- apply(training, 2, function(var){
  sum(is.na(var))/length(var)*100
})
training <- training[-which(count_nas>90)]
round(sum(is.na(training))/prod(dim(training))*100)
round(sum(is.na(testing))/prod(dim(testing))*100)
count_nas <- apply(testing, 2, function(var){
  sum(is.na(var))/length(var)*100
})
testing <- testing[-which(count_nas>90)]
round(sum(is.na(testing))/prod(dim(testing))*100)

#check if same variables are left
setdiff(names(training), names(testing))
setdiff(names(testing), names(training))

# check dimensions
dim(training); dim(testing)
#[1] 14718    53
#[1] 4904   53

#Correlation Analysis
#A correlation among variables is analysed before proceeding to the modeling procedures.
corMatrix <- cor(training[, -53])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
#Variables with high correlation are shown in dark colors in the graph above. 
#As seen in the graph, correlations are quite low, thus pca could be skipped for this assignment.

#preprocessing
preprocessModel <-preProcess(training, method=c('knnImpute', 'center', 'scale'))
pTrain <- predict(preprocessModel, training)
pTrain$classe <- training$classe
pTest <-predict(preprocessModel,testing)
pTest$classe <- testing$classe

# three methods were used for prediction : 
# Decision Tree, Random Forests, Gradient Boosting Method(gbm) 
# For each candidate model, predictions are made agaist the cross-validation data set. 
# Then, a confusion matrix is calculated and stored for each model for later reference.

#First, Decision Tree
mod_dt <- train(classe ~ ., data=pTrain, method="rpart",
                trControl = trainControl(method = "cv", 
                                         number = 4, 
                                         allowParallel = TRUE, 
                                         verboseIter = TRUE))
fancyRpartPlot(mod_dt$finalModel)
pred_dt <- predict(mod_dt, newdata = pTest)
conf_dt <- confusionMatrix(pred_dt, pTest$classe)

#Second, Random Forests
mod_rf <- train(classe ~ ., data = pTrain, method = 'rf', 
                trControl = trainControl(method = "cv", 
                                         number = 4, 
                                         allowParallel = TRUE, 
                                         verboseIter = TRUE))
pred_rf <- predict(mod_rf, newdata = pTest)
conf_rf <- confusionMatrix(pred_rf, pTest$classe)

#Third, Gradient Boosting Machine(gbm)
mod_gbm <- train(classe ~ ., data = pTrain, method = 'gbm', 
                 trControl = trainControl(method = "cv", 
                                          number = 4, 
                                          allowParallel = TRUE))
## 150 iterations were performed.
pred_gbm <- predict(mod_gbm, newdata = pTest)
conf_gbm <- confusionMatrix(pred_gbm, pTest$classe)

#Comparison of models
conf_dt$overall[1]; conf_rf$overall[1]; conf_gbm$overall[1]
#Taken together, the Random Forest model appears to be the most accurate, as expected.

##Out of Sample Error
#The out of sample error is the “error rate you get on new data set”, 
#thus is calculated as 1 - accuracy for predictions made against the cross-validation set. 
#The accuracy of the model is 0.9949. The out of sample error is 0.0051. 
#Considering that the test set is a sample size of 20, an accuracy rate well above 99% is sufficient to expect that few or none of the test samples will be mis-classified.


## Applying Selected Model to Test Set

#Same preprocessing
testSet <- test_raw[,-1:-7]
ptesting <- predict(preprocessModel, testSet)
ptesting$problem_id <- testSet$problem_id #problem_id, not classe

answers <- predict(mod_rf, ptesting)
answers <- as.character(answers)

# create function to write predictions to files
pml_write_files <- function(x) {
  n <- length(x)
  for(i in 1:n) {
    filename <- paste0("problem_id_", i, ".txt")
    write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
  }
}

# create prediction files to submit
pml_write_files(answers)
answers
# [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A" "B" "B" "B"