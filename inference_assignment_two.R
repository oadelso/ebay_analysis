# Libraries
library(tidyverse)

#load test data and review performance of model
df_test = read.csv("mse226_df_test.csv", header = TRUE)
df_test = select(df_test, -(boolean))

#run the regression on the df_test
chosen_model = lm(formula = price ~ age + kilometer + powerPS + vehicleType + brand + brand:vehicleType +
                  age:vehicleType + I(age^2):vehicleType, data = df_test)

#get RMSE values
rmse_test = sqrt(mean(predict(chosen_model, df_test) - df_test$price)**2)

#start with inference

#sample from the data
coefficients = data.frame(summary(chosen_model)$coefficients[,1])

#get the number of covariates
#r = dim(coefficients)[1]
#focus on first four for now
r = 4
n = 100
inference = matrix(rep(0, n*r), nrow = r, ncol = n)

#label each row in inference 
rownames(inference)  = rownames(covariates)[1:4]

for (i in 1:n){
  #choose with replacement indices from the df_test
  input_index = sample(1:dim(df_test)[1], dim(df_test)[1], replace = TRUE)
  
  #build a new df_test from the indices above
  df_new_test = df_test[input_index, ]
  
  #train the new model
  chosen_model = lm(formula = price ~ age + kilometer + powerPS + vehicleType + brand + brand:vehicleType +
                      age:vehicleType + I(age^2):vehicleType, data = df_new_test)
  
  #get the new value for the coefficient of the covariates
  new_coefficients = matrix( summary( chosen_model )$coefficients[1:4,1] )
  inference[ , i ] = new_coefficients
  
}

#view histogram for the values of the intercept
hist(inference[, 1])
  
