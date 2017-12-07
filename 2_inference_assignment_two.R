# Libraries
library(tidyverse)
library(ggplot2)

#load test data and review performance of model
df_test = read.csv("mse226_df_test.csv", header = TRUE)
df_cont = read.csv("mse226_df_train.csv", header = TRUE)
#remove boolean column
df_test = select(df_test, -(boolean))
df_cont = select(df_cont, -(boolean))

#run the regression on the df_cont (i.e. training set)
chosen_model = lm(formula = price ~ age + kilometer + powerPS + vehicleType + brand + brand:vehicleType +
                  age:vehicleType + I(age^2):vehicleType, data = df_cont)

#get RMSE values
rmse_test = sqrt(mean((predict(chosen_model, df_test) - df_test$price)**2))

##start with inference
#Bejamin Hochberg
#get probabilty values, and sort them
p_vals = summary(chosen_model)$coefficients[, 4]
p_vals = sort(p_vals)

#set up the values for the line alpha*j/(# of covariates)
line = 0.05*seq(1:length(p_vals))/length(p_vals)
boolean = ifelse(p_vals > line , 1, 0)

#plot 1
ggplot() + geom_line(aes(x=seq(1:285), y =line), color ='green') +
  geom_line(aes(x=seq(1:285), y = p_vals), color = 'red')

#compare by creating new vector
compare = cbind(p_vals, line, boolean)

#print out the values which satisfy the BH inequality
accept = compare[boolean == 0,]

#plot 2
ggplot() + geom_line(aes(x=seq(1:dim(accept)[1]), y =accept[,1]), color ='green') +
  +     geom_line(aes(x=seq(1:dim(accept)[1]), y = accept[,2]), color = 'red')


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
  model = lm(formula = price ~ age + kilometer + powerPS + vehicleType + brand + brand:vehicleType +
                      age:vehicleType + I(age^2):vehicleType, data = df_new_test)
  
  #get the new value for the coefficient of the covariates
  new_coefficients = matrix( summary( chosen_model )$coefficients[1:4,1] )
  inference[ , i ] = new_coefficients
  
}

#view histogram for the values of the intercept
intercept = inference[,1]
hist(intercept)
  
