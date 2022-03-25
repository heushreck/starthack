library (tidyr)
library(tidyverse)
library(dplyr)
library(tidymodels)
library(workflows)
library(tibble)
library(stringr)
library(ranger)
library(smotefamily)
library(ggfortify)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
set.seed(2022)

testmaster <- read_csv("testmaster.csv")
training_master <- read_csv("trainmaster.csv")

#-------------------------sanitizing column names-------------------------------
currentnamestrain <- colnames(training_master)
betternamestrain <- make.names(currentnamestrain)
colnames(training_master) <- betternamestrain
currentnamestest <- colnames(testmaster)
betternamestest <- make.names(currentnamestest)
colnames(testmaster) <- betternamestest
remove(currentnamestrain)
remove(betternamestrain)
remove(currentnamestest)
remove(betternamestest)

#-------------------------------------------------------------------------------
#----------------------removing unimportant columns-----------------------------

#------------------------isolating artificial test set--------------------------
picked = sample(seq_len(nrow(training_master)), size = 2716)
artificial_testmaster <- training_master[picked,]
training_master <- training_master[-picked,]
remove(picked)
#-------------------------------------------------------------------------------
#----------Getting training sets with 50/50 split of accepted/not accepted------

#1.) Using Smote
smotedata <- SMOTE(training_master[setdiff(names(training_master), "Label")], training_master$Label, K = 5, dup_size = 2)
smotedata <- smotedata$data
smotedata$class <- as.factor(smotedata$class)

#2.) Using Minority Oversampling
minoritysample <- training_master[training_master$Label == 1,]
moverdata <- rbind(training_master, minoritysample)
moverdata <- rbind(moverdata, minoritysample)
moverdata$class <- as.factor(moverdata$Label)
moverdata$Label <- NULL

#3.) Using Majority Undersampling
majoritysample <- training_master[training_master$Label == 0,]
underdata <- majoritysample[sample(nrow(majoritysample), 2036),]
underdata <- rbind(minoritysample, underdata)
underdata$class <- as.factor(underdata$Label)
underdata$Label <- NULL
remove(minoritysample)
remove (majoritysample)

#-------------------------------------------------------------------------------

#-----------------------GRID SEARCH FOR IDEAL PARAMETERS------------------------
#We now conduct a grid search where we optimize for f1 score in artificial test set
artificial_results <- artificial_testmaster$Label
artificial_testmaster$Label <- NULL

#1. smote model
hyper_grid_smote <- expand.grid(
  mtry       = seq(from = 5, to = 10, by = 1),
  sample_size = c(.632, .70, .80),
  f1score = 0,
  ntrees = seq(400, 550, by = 30)
)

#start search
print(nrow(hyper_grid_smote))
for(i in 1:nrow(hyper_grid_smote)) {
  print(i)
  
  #build model
  model_smote <- ranger(
    formula         = class ~ ., 
    data            = smotedata,
    num.trees       = hyper_grid_smote$ntrees[i],
    mtry            = hyper_grid_smote$mtry[i],
    min.node.size   = 1,
    sample.fraction = hyper_grid_smote$sample_size[i],
    seed            = 2022
  )
  #generate actual vs predictions df:
  smoteresults <- data.frame(artificial_results)
  smoteresults$actual <- smoteresults$artificial_results
  smoteresults$artificial_results <- NULL
  smoteresults$predicted <- 0
  predictions_smote <- predict(model_smote, data = artificial_testmaster)$predictions
  smoteresults$predicted <- predictions_smote
  smoteresults$predicted <- as.numeric(smoteresults$predicted)
  smoteresults$predicted <- smoteresults$predicted - 1
  
  truepositives <- 0
  truenegatives <- 0 
  falsepositives <- 0 
  falsenegatives <- 0 
  
  #calculate f1 score
  for (j in 1:nrow(smoteresults)) {
    if (smoteresults[j,]$actual == 1) {
      if (smoteresults[j,]$predicted == 1) {
        truepositives <- truepositives + 1
      }
      else {
        falsenegatives <- falsenegatives + 1
      }
    }
    else {
      if (smoteresults[j,]$predicted == 1) {
        falsepositives <- falsepositives + 1
      }
      else {
        truenegatives <- truenegatives + 1
      }
    }
  }
  precision <- truepositives / (truepositives + falsepositives)
  recall <- truepositives / (truepositives + falsenegatives)
  f1 <- 2 * ((precision * recall) / (precision + recall))
  print(f1)
  hyper_grid_smote[i,3] <- f1
}


hyper_grid_smote %>% 
  dplyr::arrange(f1score) %>%
  tail(5)

#2. over model
hyper_grid_mover <- expand.grid(
  mtry       = seq(from = 10, to = 22, by = 2),
  sample_size = c(.632, .70, .80),
  f1score = 0,
  ntrees = seq(400, 550, by = 30)
)

#start search
for(i in 1:nrow(hyper_grid_mover)) {
  print(i)
  
  #build model
  model_mover <- ranger(
    formula         = class ~ ., 
    data            = moverdata,
    num.trees       = hyper_grid_mover$ntrees[i],
    mtry            = hyper_grid_mover$mtry[i],
    min.node.size   = 1,
    sample.fraction = hyper_grid_mover$sample_size[i],
    seed            = 2022
  )
  #generate actual vs predictions df:
  moverresults <- data.frame(artificial_results)
  moverresults$actual <- moverresults$artificial_results
  moverresults$artificial_results <- NULL
  moverresults$predicted <- 0
  predictions_mover <- predict(model_mover, data = artificial_testmaster)$predictions
  moverresults$predicted <- predictions_mover
  moverresults$predicted <- as.numeric(moverresults$predicted)
  moverresults$predicted <- moverresults$predicted - 1
  
  truepositives <- 0
  truenegatives <- 0 
  falsepositives <- 0 
  falsenegatives <- 0 
  
  #calculate f1 score
  for (j in 1:nrow(moverresults)) {
    if (moverresults[j,]$actual == 1) {
      if (moverresults[j,]$predicted == 1) {
        truepositives <- truepositives + 1
      }
      else {
        falsenegatives <- falsenegatives + 1
      }
    }
    else {
      if (moverresults[j,]$predicted == 1) {
        falsepositives <- falsepositives + 1
      }
      else {
        truenegatives <- truenegatives + 1
      }
    }
  }
  precision <- truepositives / (truepositives + falsepositives)
  recall <- truepositives / (truepositives + falsenegatives)
  f1 <- 2 * ((precision * recall) / (precision + recall))
  print(f1)
  hyper_grid_mover[i,3] <- f1
}
hyper_grid_mover %>% 
  dplyr::arrange(f1score) %>%
  tail(5)

#3. under model
hyper_grid_under <- expand.grid(
  mtry       = seq(from = 4, to = 10, by = 1),
  sample_size = c(.632, .70, .80),
  f1score = 0,
  ntrees = seq(400, 550, by = 30)
)

#start search
for(i in 1:nrow(hyper_grid_under)) {
  print(i)
  
  #build model
  model_under <- ranger(
    formula         = class ~ ., 
    data            = underdata,
    num.trees       = hyper_grid_under$ntrees[i],
    mtry            = hyper_grid_under$mtry[i],
    min.node.size   = 1,
    sample.fraction = hyper_grid_under$sample_size[i],
    seed            = 2022
  )
  #generate actual vs predictions df:
  underresults <- data.frame(artificial_results)
  underresults$actual <- underresults$artificial_results
  underresults$artificial_results <- NULL
  underresults$predicted <- 0
  predictions_under <- predict(model_under, data = artificial_testmaster)$predictions
  underresults$predicted <- predictions_under
  underresults$predicted <- as.numeric(underresults$predicted)
  underresults$predicted <- underresults$predicted - 1
  
  truepositives <- 0
  truenegatives <- 0 
  falsepositives <- 0 
  falsenegatives <- 0 
  
  #calculate f1 score
  for (j in 1:nrow(underresults)) {
    if (underresults[j,]$actual == 1) {
      if (underresults[j,]$predicted == 1) {
        truepositives <- truepositives + 1
      }
      else {
        falsenegatives <- falsenegatives + 1
      }
    }
    else {
      if (underresults[j,]$predicted == 1) {
        falsepositives <- falsepositives + 1
      }
      else {
        truenegatives <- truenegatives + 1
      }
    }
  }
  precision <- truepositives / (truepositives + falsepositives)
  recall <- truepositives / (truepositives + falsenegatives)
  f1 <- 2 * ((precision * recall) / (precision + recall))
  print(f1)
  hyper_grid_under[i,3] <- f1
}

hyper_grid_under %>% 
  dplyr::arrange(f1score) %>%
  tail(5)
#-------------------------------------------------------------------------------

#----------------Generating optimal model prediction scores---------------------
#1smote
#optimal parameters found by grid search for smote: mtry 14 sampe size 0.8 ntrees 520
#generate model:
artificial_testmaster$Label <- artificial_results
training_master <- rbind(training_master, artificial_testmaster)
smotedata <- SMOTE(training_master[setdiff(names(training_master), "Label")], training_master$Label, K = 5, dup_size = 3)
smotedata <- smotedata$data
smotedata$class <- as.factor(smotedata$class)
minoritysample <- training_master[training_master$Label == 1,]
moverdata <- rbind(training_master, minoritysample)
moverdata <- rbind(moverdata, minoritysample)
moverdata <- rbind(moverdata, minoritysample)
moverdata <- rbind(moverdata, minoritysample)
moverdata$class <- as.factor(moverdata$Label)
moverdata$Label <- NULL
majoritysample <- training_master[training_master$Label == 0,]
underdata <- majoritysample[sample(nrow(majoritysample), 2036),]
underdata <- rbind(minoritysample, underdata)
underdata$class <- as.factor(underdata$Label)
underdata$Label <- NULL
remove(minoritysample)
remove (majoritysample)


ranger_smote <- ranger(
  formula = class ~ .,
  data = smotedata,
  num.trees = 520,
  mtry = 14,
  min.node.size = 1,
  sample.fraction = 0.8,
  importance = "impurity"
)

#generate predictions df:
predictions_smote <- predict(ranger_smote, data = testmaster)$predictions
submit_smote <- data.frame(testmaster$Sample_ID)
submit_smote$Label <- predictions_smote

write.csv(submit_smote,"smotemostrecent.csv", row.names = FALSE)

#2mover
#optimal parameters found by grid search for mover: mtry 7 sampe size 0.632 ntrees 500
#generate model:
ranger_mover <- ranger(
  formula = class ~ .,
  data = moverdata,
  num.trees = 430,
  mtry = 18,
  min.node.size = 1,
  sample.fraction = 0.632,
  importance = "impurity"
)

#generate predictions df:
predictions_mover <- predict(ranger_mover, data = testmaster)$predictions
submit_mover <- data.frame(testmaster$Sample_ID)
submit_mover$Label <- predictions_mover

write.csv(submit_mover,"movermostrecet.csv", row.names = FALSE)

#3under
#optimal parameters found by grid search for under: mtry 8 sample size 0.7 ntrees 420
#generate model:
ranger_under <- ranger(
  formula = class ~ .,
  data = underdata,
  num.trees = 520,
  mtry = 14,
  min.node.size = 1,
  sample.fraction = 0.8,
  importance = "impurity"
)

#generate predictions df:
predictions_under <- predict(ranger_under, data = testmaster)$predictions
submit_under <- data.frame(testmaster$Sample_ID)
submit_under$Label <- predictions_under

write.csv(submit_under,"undermostrecent.csv", row.names = FALSE)

#generate ensemble predictions

bigbosspred <- submit_smote
bigbosspred$Label <- as.numeric(bigbosspred$Label) - 1
for (k in 1:nrow(bigbosspred)) {
  bigbosspred$Label[k] <- bigbosspred$Label[k] + as.numeric(submit_under$Label[k]) - 1
  bigbosspred$Label[k] <- bigbosspred$Label[k] + as.numeric(submit_mover$Label[k]) - 1
  bigbosspred$Label[k] <- round( bigbosspred$Label[k] / 3)
}

write.csv(bigbosspred,"bossmostrecent.csv", row.names = FALSE)

var_imp_smote <- data.frame(ranger_smote$variable.importance)
var_imp_mover <- data.frame(ranger_mover$variable.importance)
var_imp_under <- data.frame(ranger_under$variable.importance)