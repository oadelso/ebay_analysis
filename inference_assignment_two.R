# Libraries
library(tidyverse)

#load test data and review performance of model

df_test = read.csv("mse226_df_test.csv", header = TRUE)

df_test = select(df_test, -(boolean))

chosen_model = lm(formula = price ~ age + kilometer + powerPS + vehicleType + brand + brand:vehicleType +
                  age:vehicleType + I(age^2):vehicleType, data = df_test)

rmse_test = sqrt(mean(predict(chosen_model, df_test) - df_test$price)**2))

#start with inference

#sample from the data
#number of simulations
covariates = data.frame(summary(chosen_model)$coefficients[,1])
#get the number of covariates
r = dim(covariates)[1]
n = 25
inference = matrix(rep(0, n*r), nrow = r, ncol = n)
#label each row in inference 
rownames(inference)  = rownames(covariates)
for (i in 1:n){
  input_index = sample(1:dim(df_test)[1], replace = TRUE)
  df_new_test = df_test[input_index, ]
  chosen_model = lm(formula = price ~ age + kilometer + powerPS + vehicleType + brand + brand:vehicleType +
                      age:vehicleType + I(age^2):vehicleType, data = df_test)
  
  new_covariates = matrix(summary(chosen_model)$coefficients[,1])
  inference[, i] = new_covariates
  
}
  
