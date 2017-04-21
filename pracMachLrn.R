# Environment Prep and Data Loading
library(caret); library(rpart); library(rpart.plot); library(rattle); library(randomForest); 
library(corrplot)

urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

trainData <- read.csv(url(urlTrain))
testCase <- read.csv(url(urlTest))

inTrain <- createDataPartition(y = trainData$classe, p = 0.7, list = FALSE)
training <- trainData[inTrain, ]
testing <- trainData[-inTrain, ]
dim(training)
dim(testing)

# Data Cleaning
# Remove Near Zero Variance variables
NZVvariables <- nearZeroVar(training)
training <- training[, -NZVvariables]
testing <- testing[, -NZVvariables]
# testCase <- testCase[, -NZVvariables]

dim(training)
dim(testing)

# Remove NA variables
# format variables as numerics
for(i in c(8:ncol(training) - 1)){
      training[, i] <- as.numeric(as.character(training[, i]))
      testing[, i] <- as.numeric(as.character(testing[, i]))
      # testCase[, i] <- as.numeric(as.character(testCase[, i]))
}

# remove NA's
goodVars <- colnames(training[colSums(is.na(training)) == 0])
goodVars <- goodVars[-c(1:6)]

training <- training[, goodVars]
testing <- testing[, goodVars]
dim(training)
dim(testing)

# Exploratory Graphs
# Correlation
corrMatrix <- cor(x = training[, -53])
corrplot(corr = corrMatrix, method = "color", type = "lower", order = "hclust", 
         hclust.method = "complete", diag = TRUE, tl.cex = 0.6, tl.col = "black")

# Prediction Models
# Decision Tree
set.seed(7769)
# modFitDT <- train(form = classe ~ ., data = training, method = "rpart")
# fancyRpartPlot(model = modFitDT$finalModel)
modDT <- rpart(formula = classe ~ ., data = training, method = "class")
fancyRpartPlot(model = modDT)

# predDT <- predict(object = modFitDT, newdata = testing)
predDT2 <- predict(object = modDT, newdata = testing, type = "class")
confuseDT <- confusionMatrix(data = predDT2, reference = testing$classe)
confuseDT$table
confuseDT$overall[1]
plot(confuseDT$table, col = confuseDT$byClass, 
     main = paste("Decision Tree Accuracy: ", round(x = confuseDT$overall[1], digits = 3)))

# Generalized Boost Model
set.seed(7769)
ctrlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM <- train(form = classe ~ ., data = training, method = "gbm", 
                   trControl = ctrlGBM, verbose = FALSE)
predGBM <- predict(object = modFitGBM, newdata = testing)
confusionGBM <- confusionMatrix(data = predGBM, reference = testing$classe)
confusionGBM$table
confusionGBM$overall[1]
plot(confusionGBM$table, col = confusionGBM$byClass, 
     main = paste("GBM Accuracy: ", round(x = confusionGBM$overall[1], digits = 3)))

# Random Forest
set.seed(7769)
modFitRF <- train(form = classe ~ ., data = training, method = "rf", 
                  trControl = trainControl(method = "cv", number = 5, verboseIter = FALSE))
predRF <- predict(object = modFitRF, newdata = testing)
confusionRF <- confusionMatrix(data = predRF, reference = testing$classe)
confusionRF
plot(confusionRF$table, col = confusionRF$byClass, 
     main = paste("Random Forest Accuracy: ", round(x = confusionRF$overall[1], digits = 3)))

# Test Case Predictions
predTESTCASE <- predict(object = modFitRF, newdata = testCase)
predTESTCASE
