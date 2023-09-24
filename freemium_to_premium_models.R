---
title: "Predictive Analysis Project - Freemium To Premium Subscribers for Music-listening Website"
author: "Allison Liu"
date: "2023-07-16"
output: html_document
---
| Workflow of building predictive models:
| 1. Import the XYZData.csv
| 2. Clean data and EDA
| 3. Cross Validation
| 4. Build the models 
| 5. Make prediction
| 6. Measure performance
| 7. Model tuning

## Import XYZData
   # Load necessary library and XYZData and summarize the data
  library(dplyr)
  library(pROC)
  library(ggplot2)
  library(caret)
  library(rpart)
  library(rpart.plot)
  library(tidymodels)
  library(tidyr)
  library(kknn)
  library(ROSE)
  library(randomForest)
  XYZData = read.csv("XYZData.csv", stringsAsFactors = TRUE)
  summary(XYZData[2:27])
   
## Clean data and EDA
   # 1. We want to check the proportion of outcome variable - adopter.
   # 2. We draw histograms for all numeric variables in one plot to understand the data distribution.
   
  XYZ_data_long <- XYZData %>%
  pivot_longer(cols = c(age, friend_cnt, avg_friend_age, avg_friend_male, friend_country_cnt, subscriber_friend_cnt, songsListened, lovedTracks, posts, playlists, shouts, tenure),
               names_to = "variable",
               values_to = "value")
   # Create a histogram for all numeric variables in one plot
   XYZ_histograms <- ggplot(XYZ_data_long, aes(x = value)) +
     geom_histogram(bins = 30, color = "black", fill = "lightblue") +
     facet_wrap(~variable, scales = "free", ncol = 4) +
     labs(title = "Histograms of Numeric Variables in the XYZDataset",
          x = "Value",
          y = "Frequency") +
     theme_minimal()
   
   # Plot the histograms
   print(XYZ_histograms)
   
   par(mfrow = c(1, 2))
   # pie chart for region
   pie(table(XYZData$adopter), labels = round(table(XYZData$adopter)/41540, 2), main ="Adopter Pie Chart", col = rainbow(3))
   legend("topright", c("0","1"), cex = 0.8, fill = rainbow(3))
   
## Split data into training and testing and oversampling
   # Randomly pick the rows for training partition
   set.seed(123)
   train_rows = createDataPartition(y = XYZData$adopter, p = 0.70, list = FALSE)
   XYZData_train = XYZData[train_rows,]
   XYZData_test = XYZData[-train_rows,]
   
   table(XYZData_train$adopter)
   table(XYZData_test$adopter)

   #Oversampling the Minority Class
   train_balanced_over <- ovun.sample(adopter ~ ., data = XYZData_train[2:27], method = "over",N = 33600)$data
   table(train_balanced_over$adopter)

## Features Selection - Information Gain
   library(FSelectorRcpp)
   IG = information_gain(adopter ~ ., data = train_balanced_over)
   # e.g., select top 5
   topK = cut_attrs(IG, k = 10)
   XYZ_topK_train = train_balanced_over %>% select(topK, adopter)
   XYZ_topK_test = XYZData_test %>% select(topK, adopter)
   XYZ_topK_train$adopter <- factor(XYZ_topK_train$adopter)
   XYZ_topK_test$adopter <- factor(XYZ_topK_test$adopter)

   
## Build the models
   1. Naive Bayes 
   2. Decision Tree
   3. Logistic Regression 
   4. K-NN
   5. Random Forest
   
   # 1. Naive Bayes Model
   library(e1071)
   library(caret)
   NB_model = naiveBayes(adopter ~ ., data = train_balanced_over, usekernel = T)
   NB_model
   # Make Predictions
   pred_nb = predict(NB_model, XYZData_test[2:27])
   prob_pred_nb = predict(NB_model, XYZData_test, type = "raw")
   XYZData_test$adopter <- factor(XYZData_test$adopter)
   confusionMatrix(data = pred_nb,
                   reference = XYZData_test$adopter,
                   mode = "prec_recall", positive = "1")
   # Make ROC and calculate AUC
   XYZ_test_roc = XYZData_test %>%
     mutate(prob = prob_pred_nb[,"1"]) %>%
     arrange(desc(prob)) %>%
     mutate(yes = ifelse(adopter=="1",1,0)) %>%
     # the following two lines make the roc curve
     mutate(TPR = cumsum(yes)/sum(yes),
            FPR = cumsum(1-yes)/sum(1-yes))
   
   ggplot(data = XYZ_test_roc, aes(x = FPR, y = TPR)) +
     geom_line() +
     theme_bw()
   # AUC
   roc_nb = roc(response = XYZ_test_roc$yes,
                predictor = XYZ_test_roc$prob)
   auc(roc_nb)

## 2. Decision Tree Model 
   # Define the number of folds for k-fold cross-validation
   set.seed(123)
   num_folds <- 5
   cv <- createFolds(XYZData$adopter, k = num_folds, list = TRUE, returnTrain = FALSE)
   # Create empty lists to store results
   confusion_matrices <- list()
   auc_scores <- numeric(num_folds)
   trees <- list()
   
   # Loop through each fold
   for (i in 1:num_folds) {
     test_rows <- cv[[i]]  # Get the test data indices for the current fold
     
     XYZ_train <- XYZData[-test_rows, ]
     XYZ_test <- XYZData[test_rows, ]
     table(XYZ_train$adopter)
     
     # Oversampling the training data - We set the total training data = 1.2(XYZ_train$adopter = 1)
     XYZ_over_train <- ovun.sample(adopter ~ ., data = XYZ_train[2:27], method = "over", N = 38400)$data
     table(XYZ_over_train$adopter)
     
     # Get the training and testing data for the current fold
     train_indices <- unlist(cv[-i])
     test_indices <- cv[[i]]
     
     train_data <- XYZ_over_train[train_indices, ]
     test_data <- XYZ_over_train[test_indices, ]
     
     # Train the model (Decision Tree)
     tree <- rpart(adopter ~ ., data = train_data,
                   method = "class",
                   parms = list(split = "information"),
                   control = rpart.control(cp = 0.0005, minsplit = 10, maxdepth = 5, minbucket = round(20/3)))
     
     # Make predictions on the test data
     pred_tree <- predict(tree, test_data, type = "class")
     test_data$adopter <- factor(test_data$adopter)
     
     # Evaluate performance using Confusion Matrix
     cm <- confusionMatrix(data = pred_tree,
                           reference = test_data$adopter,
                           mode = "prec_recall", positive = "1")
     
     # Store the confusion matrix in the list
     confusion_matrices[[i]] <- cm
     
     # Evaluate performance using ROC Curve and AUC
     pred_tree_roc <- predict(tree, test_data, type = "prob")
     roc_tree <- roc(response = test_data$adopter,
                     predictor = pred_tree_roc[, "1"])
     
     # Plot the ROC curve
     plot(roc_tree)
     
     # Calculate the AUC score
     auc_score <- auc(roc_tree)
     
     # Store the AUC score
     auc_scores[i] <- auc_score
     
     # Store the tree in the list
     trees[[i]] <- tree
     
     # Print the fold number and the corresponding confusion matrix
     cat("Fold", i, "Confusion Matrix:\n")
     print(cm)
     printcp(x = tree) 
     plotcp(x = tree)
   }
   
   # Print the AUC scores for each fold
   cat("AUC Scores:", auc_scores, "\n")
   
   # Plot the trees
   for (i in 1:length(trees)) {
     cat("Tree for Fold", i, ":\n")
     rpart.plot(trees[[i]], varlen = 0, type = 4, extra = 101, under = TRUE, cex = 0.6, box.palette = "auto")
     cat("\n")
   }
   
   # Importance variables
   library(vip)
   var_importance <- vip::vip(tree, num_features = 10)
   print(var_importance)
   
   
## 3. Logistic Regression Model
   # Define the number of folds for k-fold cross-validation
   XYZ_topK = rbind(XYZ_topK_train, XYZ_topK_test)
   num_folds <- 5
   cv <- createFolds(XYZ_topK$adopter, k = num_folds, list = TRUE, returnTrain = FALSE)
   # Create empty lists to store results
   confusion_matrices <- list()
   auc_scores <- numeric(num_folds)
   
   # Loop through each fold
   for (i in 1:num_folds) {
     test_rows <- cv[[i]]  # Get the test data indices for the current fold
     
     XYZ_train <- XYZ_topK[-test_rows, ]
     XYZ_test <- XYZ_topK[test_rows, ]
     #table(XYZ_topK$adopter)
     
     # Oversampling the training data - We set the total training data = 1.2(XYZ_train$adopter = 1)
     XYZ_over_train <- ovun.sample(adopter ~ ., data = XYZ_train, method = "over", N = 38400)$data
     #table(XYZ_over_train$adopter)
     
     # Get the training and testing data for the current fold
     train_indices <- unlist(cv[-i])
     test_indices <- cv[[i]]
     
     train_data <- XYZ_over_train[train_indices, ]
     test_data <- XYZ_over_train[test_indices, ]
     
     # Train the model (Logistic Regression)
     train_data$adopter <- factor(train_data$adopter)
     
     mylogit <- glm(adopter ~., data = train_data, family = "binomial")
     
     # Make predictions on the test data
     pred_log = predict(mylogit, test_data, type = "response")
     pred_cat <- ifelse(pred_log >= 0.5, 1, 0)
     
     # Evaluate performance using Confusion Matrix
     cm = confusionMatrix(factor(pred_cat), factor(test_data$adopter), 
                          mode = "prec_recall", positive = '1')
     
     # Store the confusion matrix in the list
     confusion_matrices[[i]] <- cm
     
     # Evaluate performance using ROC Curve and AUC
     pred_log_roc = predict(mylogit, XYZData_test, type = "response")
     roc_log = roc(response = XYZData_test$adopter,
                   predictor = pred_log_roc)
     
     # Plot the ROC curve
     plot(roc_log)
     
     # Calculate the AUC score and Store it
     auc_score <- auc(roc_log)
     auc_scores[i] <- auc_score
     
     
     # Print the fold number and the corresponding confusion matrix
     cat("Fold", i, "Confusion Matrix:\n")
     print(cm)
   }
   
   # Print the AUC scores for each fold
   cat("AUC Scores:", auc_scores, "\n")
   summary(mylogit)
   
  
## 4. K-NN
   # Normalize
   normalize = function(x){
     return ((x - min(x))/(max(x) - min(x)))
   }
   XYZ_normalized = XYZData %>% mutate_at(2:26, normalize)
   XYZ_normalized_train = XYZ_normalized[train_rows,]
   XYZ_normalized_test = XYZ_normalized[-train_rows,]
   
   # Oversampling
   XYZ_train_oversampling <- ovun.sample(adopter ~ ., data = XYZ_normalized_train, 
                                         method = "over", N = 36000)$data
   
   # Feature Selection: Filter Approach
   IG = information_gain(adopter ~ ., data = XYZ_train_oversampling)
   # select top 10
   topK = cut_attrs(IG, k = 10)
   XYZ_topK_train = XYZ_train_oversampling %>% select(topK, adopter, user_id)
   XYZ_topK_test = XYZ_normalized_test %>% select(topK, adopter, user_id)
   
   # KNN
   XYZ_topK_train$adopter <- factor(XYZ_topK_train$adopter)
   XYZ_topK_test$adopter <- factor(XYZ_topK_test$adopter)
   
   mod_knn = train(x = XYZ_topK_train,
                   y = XYZ_topK_train$adopter, 
                   method = "knn",
                   preProcess = c("center", "scale"),
                   trControl = trainControl(method = "repeatedcv",
                                            number = 10,
                                            repeats = 3)
   )
   # View the fitted model.
   mod_knn
   plot(mod_knn)

   # Make Prediction and Evaluated the Model
   set.seed(123)
   pred_cls = predict(mod_knn, XYZ_topK_test, k = 5)
   head(pred_cls, 50)
   
   confusionMatrix(data = pred_cls,
                   reference = XYZ_topK_test[,11],
                   mode = "prec_recall", positive = "1")
   
## 5. Random Forest
   #Run the random forest model
   set.seed(123)
   rf <-randomForest(adopter~.,data=train_balanced_over, ntree=500) 
   print(rf)
   
   # Prediction and Confusion Matrix - Test Data
   p2 <- predict(rf, XYZData_test[2:27])
   confusionMatrix(p2, XYZData_test$adopter, positive = '1')
   
   plot(rf)
   
   #Select mtry value with minimum out of bag(OOB) error.
   mtry <- tuneRF(train_balanced_over[,-26], train_balanced_over[,26],
                  stepFactor = 0.5,
                  plot = TRUE,
                  ntreeTry = 150,
                  trace = TRUE,
                  improve = 0.05)
   best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
   print(mtry)
   print(best.m)
   
   # Number of nods for the trees
   hist(treesize(rf),
        main = "No. of Nodes for the Trees",
        col = "green")
   # Variable Importance
   varImpPlot(rf,
              sort = T,
              n.var = 10,
              main = "Top 10 - Variable Importance")
   importance(rf)
   
   # Multi-dimensional Scaling Plot of Proximity Matrix
   ctrl <- trainControl(method = "cv",  
                        number = 10, 
                        verboseIter = TRUE)
   rf_model <- train(adopter ~ ., data = train_balanced_over,
                     method = "rf",      
                     trControl = ctrl)   
   print(rf_model)
      
