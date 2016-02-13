#Practical Machine Learning – Final Project Write-Up
The goal of this project was to develop an algorithm to predict how well participants perform a particular physical activity using information about how the activity was executed. Specifically, the aim was to predict how well participants executed barbell lifts using data from accelerometers on the belt, forearm, arm, and dumbbell. There were five possible outcomes: the lift was performed exactly according to the specification (Class A), the participant threw his or her elbows to the front (Class B), the participant lifted the dumbbell only halfway (Class C), the participant lowered the dumbbell only halfway (Class D,) and and the participant threw his or her hips to the front (Class E).


##Data Exploration & Cleaning
Two sets of data were made available for this project: a training set consisting of 160 variables and 19,622 cases and a test set consisting of the same variables but only 20 cases.  (See http://groupware.les.inf.puc-rio.br/har for more information about the data and how they were obtained.)
First, I read the training data and noted several missing values on many of the predictor variables.

	mydata <-read.csv("~/pml-training.csv")
	dim(mydata)
	View(mydata)

I removed variables that had one or more missing values in the training set, and I removed variables that were neither physical parameters to use for prediction nor the outcome variable I wanted to predict (the “classe” of the activity—A, B, C, D, or E). I performed the same manipulations on the test set so it would mirror the training set.

	mydata <- mydata[c(2:11,37:49,60:68,84:86,102,113:124,140,151:160)]
	testdata <- testdata[c(2:11,37:49,60:68,84:86,102,113:124,140,151:160)]
	mydata <- mydata[-c(1:6)]
	testdata <- testdata[-c(1:6)]

I reviewed the structure of the training data and determined I could move forward with developing a prediction algorithm.

	str(mydata)

My final training and testing sets comprised 52 predictors and 1 outcome variable (“classe”)

##Data Partitioning
Next, I separated the training data into two parts: a training set and a hold-out sample to test the algorithm built using the training set. I randomly assigned three-quarters of the training data to be the training set and the remaining quarter to be the test set.
	
	set.seed(814)
	inTrain <- createDataPartition(y=mydata$classe,p=.75,list=FALSE)
	training <- mydata[inTrain,]
	testing <- mydata[-inTrain,]

The training set comprised 14,718 cases while the test set comprised 4,904.


##Model Construction & Results
In my first attempt, I used linear discriminant analysis with 5-fold cross-validation in my first attempt to develop an algorithm that would accurately classify cases into the five possible “classe” categories. I used LDA because this is an analytic approach with which my colleagues are familiar.

	library(caret)
	mycontrol <- trainControl(method="cv", number=5, allowParallel=TRUE, verbose=TRUE)
	modFitlda <- train(classe ~ .,method="lda",data=training, trControl=mycontrol, verbose=FALSE)

The train function was executed very quickly. However, accuracy suffered substantially.

	modFitlda

	Linear Discriminant Analysis 

	14718 samples
   	52 predictor
	5 classes: 'A', 'B', 'C', 'D', 'E' 

	No pre-processing
	Resampling: Cross-Validated (5 fold) 
	Summary of sample sizes: 11775, 11773, 11776, 11772, 11776 
	Resampling results

  	Accuracy   Kappa     Accuracy SD  Kappa SD   
	0.6996201  0.619922  0.005049872  0.006399358

Within the training set, accuracy was less than 70%. When applied to the test set, accuracy was even lower at 71% (see confusion matrix below).

	predlda    A    B    C    D    E
      	A 1158  132   84   61   30
      	B   37  620   74   28  159
      	C   97  116  574   84   79
      	D   97   32   99  603   98
      	E    6   49   24   28  535

Due to low accuracy, I decided to implement random forest instead, again using 5-fold cross-validation in an effort to avoid overfitting.

	modFitrf <- train(classe ~ .,method="rf",data=training,trControl=mycontrol, verbose=FALSE)

	Random Forest 

	14718 samples
   	52 predictor
    	5 classes: 'A', 'B', 'C', 'D', 'E' 

	No pre-processing
	Resampling: Cross-Validated (5 fold) 
	Summary of sample sizes: 11775, 11773, 11776, 11772, 11776 
	Resampling results across tuning parameters:

  	mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
   	2    0.9924571  0.9904581  0.002573112  0.003255982
  	27    0.9921179  0.9900289  0.002249167  0.002845970
  	52    0.9845767  0.9804874  0.002004701  0.002535956

	Accuracy was used to select the optimal model using  the largest value.
	The final value used for the model was mtry = 2. 

This approach proved much more accurate than my initial attempt with LDA, with zero classification errors in training.
	predrf    A    B    C    D    E
     	A 4185    0    0    0    0
     	B    0 2848    0    0    0
     	C    0    0 2567    0    0
     	D    0    0    0 2412    0
     	E    0    0    0    0 270

When I applied the model to the test set, accuracy was still extremely high at 99%.
	predrf    A    B    C    D    E
     	A 1395    5    0    0    0
     	B    0  942    7    0    0
     	C    0    2  848   15    1
     	D    0    0    0  788    3
     	E    0    0    0    1  897

The confusion matrix shows there were only 32 classification errors of the total 4,904 cases in the test set.
In sum, the final model built using random forest with 5-fold cross-validation was highly accurate with an expected out-of-sample error rate of less than 5%, which justified the cost in terms of speed.
