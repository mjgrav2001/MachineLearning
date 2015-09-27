#-------------------------------------------------------------------------
# This R script is called 'run_analysis_ML.R'. It provides the analysis for the course project 
# in the 'Practical Machine Learning' course for the Coursera course series 'Data Science' 
# by Johns Hopkins University. 
#
# Note: The data linked to the course website can be found here:
# https://github.com/mjgrav2001/MachineLearning
#
# The analysis refers to following publication:
#
# Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H.:
# Wearable Computing: Accelerometers' Data Classification of Body Postures 
# and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. 
# Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science.,
# pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. 
# DOI: 10.1007/978-3-642-34459-6_6. 
#
# Additional information is available online at: 
# http://groupware.les.inf.puc-rio.br/har#ixzz3lqZ2yZHW
#-------------------------------------------------------------------------
# 
# Following information is provided about the available data by the authors at the 
# website listed above:
#
# "... Six young health participants were asked to perform one set of 10 
# repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: 
# exactly according to the specification (Class A), throwing the elbows to the front 
# (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only 
# halfway (Class D) and throwing the hips to the front (Class E). ..."
#
#-------------------------------------------------------------------------
# Loading necessary libraries in R:
library(plyr)
library(dplyr)
library(Hmisc)
library(lubridate)
library(lattice)
library(ggplot2)
library(caret)
library(AppliedPredictiveModeling)
library(ElemStatLearn)
library(stats)
library(pgmm)
library(rpart)
library(gbm)
library(mda)
library(RWeka)
library(forecast)
library(e1071)
library(randomForest)
library(mgcv)
library(nlme)
library(glmnet)
library(lars)
library(genlasso)
library(elasticnet)
#-------------------------------------------------------------------------
# Creating, cleaning and preparing training and testing data sets:
alldata <- read.csv("./data/pml-training.csv", na.strings = c("NA", "#DIV/0!", "", " "), stringsAsFactors=F)
testing <- read.csv("./data/pml-testing.csv", na.strings = c("NA", "#DIV/0!", "", " "), stringsAsFactors=F)
#
inTrain = createDataPartition(alldata$classe, p = 0.6)[[1]]
training = alldata[ inTrain,]
validation = alldata[-inTrain,]
#
dim(training)
dim(validation)
dim(testing)
training <- training[,colSums(is.na(training)) == 0]
validation <- validation[,colSums(is.na(validation)) == 0]
testing <- testing[,colSums(is.na(testing)) == 0]
str(training)
str(testing)
#
#-------------------------------------------------------------------------
# Removing descriptors that describe time stamp information or integrated kinematic variables
#
myvars <- names(training) %in% c("X", "user_name", "cvtd_timestamp", "new_window", 
                                 "raw_timestamp_part_1", "raw_timestamp_part_2", "num_window", 
                                 "total_accel_belt", "total_accel_arm", "total_accel_dumbbell", 
                                 "total_accel_forearm") 
training <- training[!myvars]
validation <- validation[!myvars]
testing <- testing[!myvars]
#
all.equal(names(training),names(validation))
all.equal(names(training),names(testing))
#
#-------------------------------------------------------------------------
# Transform variable 'classe' into factor variable
#
training$classe <- factor(training$classe) 
validation$classe <- factor(validation$classe) 
testing$problem_id <- factor(testing$problem_id)
#
#-------------------------------------------------------------------------
# Create second data sets 'training0', 'validation0', 'testing0' with variable 'classe' removed for later model fits
#
training0 <- training[c(-49)]
validation0 <- validation[c(-49)]
testing0 <- testing[c(-49)]
#
#-------------------------------------------------------------------------
# Removing highly correlated descriptors (i.e. descriptors that are more than 75% correlated) in all data sets
# for later model fits to prevent overfitting
#
descrCor <- cor(training0)
summary(descrCor[upper.tri(descrCor)])
#
highlyCorDescr <- findCorrelation(descrCor, cutoff = .75)
training_filt <- training0[,-highlyCorDescr]
descrCor2 <- cor(training_filt)
summary(descrCor2[upper.tri(descrCor2)])
#
validation_filt <- validation0[,-highlyCorDescr]
testing_filt <- testing0[,-highlyCorDescr]
#
dim(training0)
dim(training_filt)
dim(validation_filt)
dim(testing_filt)
#
# Checking for variables with near zero variance:
nzv <- nearZeroVar(training_filt, saveMetrics = TRUE)
nzv
#
#-------------------------------------------------------------------------
# Start with a Principal Component Analysis (PCA) to determine the number of relevant variables:
#
set.seed(3433)
preProc <- preProcess(training_filt, method = "pca", pcaComp = 20)
trainPC <- predict(preProc, training_filt)
summary(prcomp(trainPC))
modelFit <- train(factor(training$classe) ~ ., method = "rf", data = trainPC)
testPC <- predict(preProc, validation_filt)
confusionMatrix(predict(modelFit, testPC), factor(validation$classe))
preProc <- preProcess(training_filt, method = "pca", pcaComp = 25)
trainPC <- predict(preProc, training_filt)
summary(prcomp(trainPC))
modelFit <- train(factor(training$classe) ~ ., method = "rf", data = trainPC)
testPC <- predict(preProc, validation_filt)
#
confusionMatrix(predict(modelFit, trainPC), factor(training$classe))
confusionMatrix(predict(modelFit, testPC), factor(validation$classe))
#
#-------------------------------------------------------------------------
# Perform individual model fits to all remaining 33 variables with variable 'classe' as outcome
#
# Multiple models:
set.seed(32323)
fitControl <- trainControl(method = "none")
tgrid     <- expand.grid(mtry=c(6)) 
#
modFit1 <- train(training$classe ~ ., method="rf", trControl = fitControl, tuneGrid = tgrid, data=training_filt, prox=TRUE)
pred1 <- predict(modFit1, newdata = validation_filt)
confusionMatrix(factor(pred1), factor(validation$classe))
#
modFit2 <- train(training$classe ~ ., method="gbm", data=training_filt, verbose=FALSE)
pred2 <- predict(modFit2, newdata = validation_filt)
confusionMatrix(factor(pred2), factor(validation$classe))
#
modFit3 <- svm(formula = training$classe ~ ., data=training_filt)
pred3 <- predict(modFit3, newdata = validation_filt)
confusionMatrix(factor(pred3), factor(validation$classe))
#
modFit4 <- train(training$classe ~ ., method="tree", data=training_filt)
pred4 <- predict(modFit4, newdata = validation_filt)
confusionMatrix(factor(pred4), factor(validation$classe))
#
#modFit5 <- train(training$classe ~ ., method="gam", data=training_filt)
#pred5 <- predict(modFit5, newdata = validation_filt)
#confusionMatrix(factor(pred5), factor(validation$classe))
#
modFit6 <- train(training$classe ~ ., method="mda", data=training_filt)
pred6 <- predict(modFit6, newdata = validation_filt)
confusionMatrix(factor(pred6), factor(validation$classe))
#
modFit7 <- train(training$classe ~ ., method="glm", data=training_filt)
pred7 <- predict(modFit7, newdata = validation_filt)
confusionMatrix(factor(pred7), factor(validation$classe))
#
#-------------------------------------------------------------------------
# Combined model fit:
pred_data123 <- data.frame(pred1, pred2, pred3)
combo_modFit123 <- train(factor(validation$classe) ~ ., method="rf", trControl = fitControl, 
                         tuneGrid = tgrid, data = pred_data123)
combo_pred123 <- predict(combo_modFit123, newdata = validation_filt)
confusionMatrix(combo_pred123, factor(validation$classe))
#
pred1T <- predict(modFit1, newdata = testing_filt)
pred2T <- predict(modFit2, newdata = testing_filt)
pred3T <- predict(modFit3, newdata = testing_filt)
pred123T <- data.frame(pred1T, pred2T, pred3T)
combo_modFit123T <- train(factor(validation$classe) ~ ., method="rf", trControl = fitControl, 
                         tuneGrid = tgrid, data = pred123T)
combo_pred123T <- predict(combo_modFit123T, newdata = pred123T)
confusionMatrix(combo_pred123T, factor(validation$classe))                      
#
#-------------------------------------------------------------------------
# Generate text files for test cases:
answers1 <- predict(modFit1, newdata = testing_filt)
answers2 <- predict(modFit2, newdata = testing_filt)
answers3 <- predict(modFit3, newdata = testing_filt)
answers1
answers2
answers3
answers <- answers2
for (i in 1:20) {
  if ((answers3[i] == answers2[i]) || (answers3[i] == answers1[i])) {
    answers[i] = answers1[i]
  }  
}
answers 
#
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("/Users/markjack/data/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,
                        row.names=FALSE,col.names=FALSE)
  }
}
#
pml_write_files(answers)
#
# Plots:

# Scatter Plots
transparentTheme(trans = .4)

featurePlot(x = training_filt[, c("magnet_forearm_x", "magnet_forearm_z")],
            y = factor(training$classe),
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))

featurePlot(x = training_filt[, c("yaw_belt", "roll_arm", "pitch_arm", "yaw_arm", "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell", "roll_forearm", "pitch_forearm", "yaw_forearm", "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z")],
            y = factor(training$classe),
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 12))

featurePlot(x = training_filt[, c("roll_arm", "pitch_arm", "yaw_arm")],
            y = factor(training$classe),
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))

featurePlot(x = training_filt[, c("roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell")],
            y = factor(training$classe),
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))

featurePlot(x = training_filt[, c("roll_forearm", "pitch_forearm", "yaw_forearm")],
            y = factor(training$classe),
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))

featurePlot(x = training_filt[, c("magnet_forearm_x", "magnet_forearm_z")],
            y = factor(training$classe),
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 2))

featurePlot(x = training_filt[, c("yaw_belt", "gyros_arm_y", "gyros_arm_z",
                                  "gyros_belt_x", "gyros_belt_y", "gyros_belt_z",
                                  "gyros_dumbbell_y", "gyros_forearm_x", "gyros_forearm_z")],
            y = factor(training$classe),
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 9))

# Box Plots
transparentTheme(trans = .5)

featurePlot(x = training_filt[, c("roll_arm", "pitch_arm", "yaw_arm", 
                                  "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell",
                                  "roll_forearm", "pitch_forearm", "yaw_forearm",
                                  "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z")],
            y = factor(training$classe),
            plot = "box",
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),
            layout = c(3, 4),
            auto.key = list(columns = 3))

featurePlot(x = training_filt[, c("roll_arm", "pitch_arm", "yaw_arm", 
                                  "roll_dumbbell", "pitch_dumbbell", "yaw_dumbbell",
                                  "roll_forearm", "pitch_forearm", "yaw_forearm")],
            y = factor(training$classe),
            plot = "box",
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),
            layout = c(3, 3),
            auto.key = list(columns = 3))

featurePlot(x = training_filt[, c("gyros_belt_x", "gyros_belt_y", "gyros_belt_z",
                                  "gyros_dumbbell_x", "gyros_dumbbell_y", "gyros_dumbbell_z",
                                  "gyros_forearm_x", "gyros_forearm_y", "gyros_forearm_z", 
                                  "gyros_arm_y", "gyros_arm_z")],
            y = factor(training$classe),
            plot = "box",
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),
            layout = c(3, 4),
            auto.key = list(columns = 3))

featurePlot(x = training_filt[, c("accel_belt_y", "accel_arm_z", "accel_dumbbell_y",
                                  "accel_forearm_x", "accel_forearm_z", "yaw_belt")],
            y = factor(training$classe),
            plot = "box",
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),
            layout = c(3, 2),
            auto.key = list(columns = 3))

featurePlot(x = training_filt[, c("magnet_belt_x", "magnet_belt_y", "magnet_dumbbell_z", "magnet_forearm_x", "magnet_forearm_y", "magnet_forearm_z", "magnet_arm_x")],
            y = factor(training$classe),
            plot = "box",
            ## Pass in options to bwplot() 
            scales = list(y = list(relation="free"),
                          x = list(rot = 90)),
            layout = c(3, 3),
            auto.key = list(columns = 3))

# Scatter Plot of Most Variable Data
featurePlot(x = training_filt[, c("yaw_belt", "accel_dumbbell_y",
                                  "accel_forearm_x",
                                  "roll_dumbbell", "yaw_dumbbell",
                                  "roll_forearm", "yaw_forearm",
                                  "magnet_arm_x")],
            y = factor(training$classe),
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 8))

# Scatter Plot of Most Variable Data
featurePlot(x = training_filt[, c("yaw_belt", "accel_dumbbell_y",
                                  "accel_forearm_x",
                                  "magnet_arm_x")],
            y = factor(training$classe),
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 4))

featurePlot(x = training_filt[, c("yaw_belt", 
                                  "roll_dumbbell", "yaw_dumbbell",
                                  "roll_forearm", "yaw_forearm")],
            y = factor(training$classe),
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 5))

featurePlot(x = training_filt[, c("yaw_belt", 
                                  "roll_dumbbell", "yaw_dumbbell")],
            y = factor(training$classe),
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))

featurePlot(x = training_filt[, c("yaw_belt", 
                                  "roll_forearm", "yaw_forearm")],
            y = factor(training$classe),
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 3))


