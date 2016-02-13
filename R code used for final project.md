#R code for final PLM project

##load required packages

  library(caret)

##read and examine training and test data

  mydata <-read.csv("~/pml-training.csv")
  
  dim(mydata)
  
  View(mydata)
  
  summary(mydata$classe)
  

  testdata <- read.csv("~/pml-testing.csv") #see if testing data set is the same

##remove vars that are missing in training set (and do the same for testing set)

  mydata <- mydata[c(2:11,37:49,60:68,84:86,102,113:124,140,151:160)]
  
  testdata <- testdata[c(2:11,37:49,60:68,84:86,102,113:124,140,151:160)]

##remove vars that are not physical parameters (user_name, timestamps, etc.)

  mydata <- mydata[-c(1:6)]
  
  testdata <- testdata[-c(1:6)]
  
  str(mydata)

##create training and testing data frames

  set.seed(814)
  
  inTrain <- createDataPartition(y=mydata$classe,p=.75,list=FALSE)
  
  training <- mydata[inTrain,]
  
  testing <- mydata[-inTrain,]


##fit linear discriminant analysis

  mycontrol <- trainControl(method="cv", number=5, allowParallel=TRUE, verbose=TRUE)
  
  modFitlda <- train(classe ~ .,method="lda",data=training, trControl=mycontrol, verbose=FALSE)
  
  modFitlda #look at the model fit, including accuracy
  
  predlda <- predict(modFitlda,newdata=training)
  
  table(predlda,training$classe) #examine the confusion matrix for the training set
  
  predlda <- predict(modFitlda,newdata=testing)
  
  table(predlda,testing$classe) #examine the confusion matrix for the testing set
  

##fit random forest

  set.seed(814)
  
  modFitrf <- train(classe ~ .,method="rf",data=training,trControl=mycontrol, verbose=FALSE)
  
  modFitrf #look at the model fit, including accuracy
  
  predrf <- predict(modFitrf,newdata=training)
  
  table(predrf,training$classe) #examine the confusion matrix for the training set
  
  predrf <- predict(modFitrf,newdata=testing)
  
  table(predrf,testing$classe) #examine the confusion matrix for the testing set
  


##predict classe for 20 cases in testdata dataframe

  predrf <- predict(modFitrf,newdata=testdata)
  
  predrf
