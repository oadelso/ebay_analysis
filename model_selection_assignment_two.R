# ============================================================================================= #
## Set-up
# Working directory & removing memory
# ============================================================================================= #
rm(list = objects())

# Libraries
library(tidyverse)
library(tibble)
library(ggplot2)
library(glmnet)
library(standardize)
library(leaps)
library(ROCR)

# Read data
df = read.csv("autos.csv", header = TRUE, colClasses = c("character", "character", "character",
                                                         "character", "numeric", "character",
                                                         "character", "numeric", "character",
                                                         "numeric", "character", "numeric",
                                                         "numeric", "character", "character",
                                                         "character", "character", "numeric",
                                                         "numeric", "character"), fileEncoding="latin1")


# Seller Translation
df$seller[df$seller == "privat"] = "Private"
df$seller[df$seller == "gewerblich"] = "Commercial"

# Offer Type Translation
df$offerType[df$offerType == "Angebot"] = "Offer"
df$offerType[df$offerType == "Gesuch"] = "Demand"

# Vehicle Type Translation
df$vehicleType[df$vehicleType == ""] = "Other"
df$vehicleType[df$vehicleType == "andere"] = "Other"
df$vehicleType[df$vehicleType == "bus"] = "Bus"
df$vehicleType[df$vehicleType == "cabrio"] = "Spider"
df$vehicleType[df$vehicleType == "coupe"] = "Coupe"
df$vehicleType[df$vehicleType == "kleinwagen"] = "Small"
df$vehicleType[df$vehicleType == "kombi"] = "Station"
df$vehicleType[df$vehicleType == "limousine"] = "Sedan"
df$vehicleType[df$vehicleType == "suv"] = "SUV"

# Fuel Translation
df$fuelType[df$fuelType == ""] = "Other"
df$fuelType[df$fuelType == "andere"] = "Other"
df$fuelType[df$fuelType == "diesel"] = "Diesel"
df$fuelType[df$fuelType == "benzin"] = "Gasoline"
df$fuelType[df$fuelType == "cng"] = "NaturalGas"
df$fuelType[df$fuelType == "elektro"] = "Electric"
df$fuelType[df$fuelType == "hybrid"] = "Hybrid"
df$fuelType[df$fuelType == "lpg"] = "LPG"

# Gearbox Translation
df$gearbox[df$gearbox == ""] = "Other"
df$gearbox[df$gearbox == "manuell"] = "Manual"
df$gearbox[df$gearbox == "automatik"] = "Automatic"

# Not Repaired Damage Translation
df$notRepairedDamage[df$notRepairedDamage == ""] = "No"
df$notRepairedDamage[df$notRepairedDamage == "ja"] = "Yes"
df$notRepairedDamage[df$notRepairedDamage == "nein"] = "No"

# Model
df$model[df$model == ""] = "Other"
# ============================================================================================= #

# Creating new variables
# Creating a new month registration for all cars without one
df$monthOfRegistration[df$monthOfRegistration == ""] = 12
df$monthOfRegistration[df$monthOfRegistration == 0] = 12
# Computing age in years
df["age"] = (12 * (2018 - df$yearOfRegistration) + 1 - df$monthOfRegistration)/12

# New boolean variable
# Identify the last day of crawling
df$dateCrawled = as.Date(df$dateCrawled, "%Y-%m-%d")
df$lastSeen = as.Date(df$lastSeen, "%Y-%m-%d")
df$dateCreated = as.Date(df$dateCreated, "%Y-%m-%d")
finalDay = max(df$dateCrawled)

# Two temporary columns will be built to help with this step
# Number of days between last seen and the final date of crawling
df$addCol1 = finalDay - df$lastSeen
# Nnumber of days between created and final date of crawling
df$addCol2 = df$lastSeen - df$dateCreated
# Define the new column
df$boolean = 0
# Set to 1 all the rows that satisfy the conditions
df$boolean[df$addCol1 > 1 & df$addCol1 < 30] = 1

# Filter data
df = filter(df, yearOfRegistration > 1950 & yearOfRegistration < 2018)
df = filter(df, powerPS > 20 & powerPS < 801)
df = filter(df, price < 300000 & price > 1000)


df = select(df, price, brand, kilometer, age, powerPS, gearbox, vehicleType, fuelType, 
            seller, notRepairedDamage, boolean) #postalCode, model,boolean)

## Prediction task
# Train/Test
set.seed(1)
indices = sample(1:dim(df)[1])
# Leave 20% aside
index_test = round(dim(df)[1] * 0.2)
df_test = df[indices[1:index_test],]
# Keep 80% for training
df_train_tmp = df[indices[(index_test+1):length(indices)],]

indices = sample(1:dim(df_train_tmp)[1])
# Leave 7% aside
index_val = round(dim(df)[1] * 0.07)
df_val = df[indices[1:index_val],]
# Keep 93% for training
df_train = df[indices[(index_val + 1):length(indices)],]

# Classification
# Datasets
df_cont = select(df_train, -c(boolean))
df_cont_val = select(df_val, -c(boolean))

# Dumb baseline
mean_price = mean(df_cont$price)
rmse_dumb = sqrt(mean((mean_price - df_cont_val$price)**2))
max(df_cont$price)

#remove 'offerType from df_cont
df_cont=select(df_cont,-(offerType))
# Baseline (all covariates)
model= lm(price~., data = df_cont)
summary(model)
rmse_base = sqrt(mean((redict(model, df_cont_val) - df_cont_val$price)**2))

#step wise selection
model_min=lm(price ~ 1, data = df_cont)
model_max= lm(price~., data = df_cont)
step(model_min, scope=list(lower=model_min, upper=model_max), direction="forward")
chosen_model=lm(formula = price ~ powerPS + kilometer + brand + fuelType + 
                  +                     vehicleType + age + notRepairedDamage + gearbox, data = df_cont)

# Lasso
df_cont_lasso = df_cont
df_cont_lasso$price = (df_cont$price - mean(df_cont$price))/sd(df_cont$price)
df_cont_lasso$kilometer = (df_cont$kilometer - mean(df_cont$kilometer))/sd(df_cont$kilometer)
df_cont_lasso$age = (df_cont$age - mean(df_cont$age))/sd(df_cont$age)
df_cont_lasso$powerPS = (df_cont$powerPS - mean(df_cont$powerPS))/sd(df_cont$powerPS)

df_cont_val_lasso = df_cont_lasso[indices[1:index_val],]

X = select(df_cont, -c(price))
Y = df_cont$price

model_lasso = glmnet(X, Y, alpha = 1, lambda = 1000)
summary(model_lasso)

#tested models
#interaction terms (0.5681)
chosen_model=lm(formula = price ~ powerPS + kilometer + brand + fuelType + 
                  vehicleType + age + notRepairedDamage + gearbox + powerPS:fuelType, data = df_cont)

#interaction terms (0.5692)
chosen_model=lm(formula = price ~ powerPS + kilometer + brand + fuelType + 
                  vehicleType + age + notRepairedDamage + gearbox + powerPS:fuelType + age:kilometer, data = df_cont)

#interacton term (0.5705)
chosen_model=lm(formula = price ~ powerPS + kilometer + brand + fuelType + 
                  vehicleType + age + notRepairedDamage + gearbox + powerPS:fuelType + age:kilometer + gearbox:fuelType, data = df_cont)

#interaction term (0.5725)
chosen_model=lm(formula = price ~ powerPS + kilometer + brand + fuelType + 
                  vehicleType + age + notRepairedDamage + gearbox + powerPS:fuelType + age:kilometer + gearbox:fuelType + brand:notRepairedDamage, data = df_cont)

#interaction term (0.5728)
chosen_model=lm(formula = price ~ powerPS + kilometer + brand + fuelType + 
                  vehicleType + age + notRepairedDamage + gearbox + 
                  powerPS:fuelType + age:kilometer + gearbox:fuelType + 
                  brand:notRepairedDamage + vehicleType:notRepairedDamage, data = df_cont)

#interaction term (0.5764)
chosen_model=lm(formula = price ~ powerPS + kilometer + brand + fuelType + 
                  vehicleType + age + notRepairedDamage + gearbox + powerPS:fuelType + 
                  age:kilometer + gearbox:fuelType + brand:notRepairedDamage + 
                  vehicleType:notRepairedDamage + gearbox:kilometer, data = df_cont)

#interaction term (0.5864)
chosen_model=lm(formula = price ~ powerPS + kilometer + brand + fuelType + 
                  vehicleType + age + notRepairedDamage + gearbox + powerPS:fuelType + 
                  age:kilometer + gearbox:fuelType + brand:notRepairedDamage + vehicleType:notRepairedDamage + 
                  gearbox:kilometer, data = df_cont)

#interaction terms (0.5892)
chosen_model=lm(formula = price ~ powerPS + kilometer + brand + fuelType + 
                  vehicleType + age + notRepairedDamage + gearbox + powerPS:fuelType + 
                  age:kilometer + gearbox:fuelType + brand:notRepairedDamage + vehicleType:notRepairedDamage + 
                  gearbox:kilometer + gearbox:age, data = df_cont)

#interaction terms (0.6149)
chosen_model=lm(formula = price ~ powerPS + kilometer + brand + fuelType + 
                  vehicleType + age + notRepairedDamage + gearbox + powerPS:fuelType + 
                  age:kilometer + gearbox:fuelType + brand:notRepairedDamage + vehicleType:notRepairedDamage + 
                  gearbox:kilometer + gearbox:age +brand:vehicleType, data = df_cont)

#interaction terms (0.6203)
chosen_model=lm(formula = price ~ powerPS + kilometer + brand + fuelType + 
                  vehicleType + age + notRepairedDamage + gearbox + powerPS:fuelType + 
                  age:kilometer + gearbox:fuelType + brand:notRepairedDamage + 
                  vehicleType:notRepairedDamage + gearbox:kilometer + gearbox:age + 
                  brand:vehicleType + brand:gearbox, data = df_cont)

#interaction terms (0.6209)
chosen_model=lm(formula = price ~ powerPS + kilometer + brand + fuelType + 
                  vehicleType + age + notRepairedDamage + gearbox + powerPS:fuelType + 
                  age:kilometer + gearbox:fuelType + brand:notRepairedDamage + vehicleType:notRepairedDamage + 
                  gearbox:kilometer + gearbox:age + brand:vehicleType + brand:gearbox + 
                  brand:gearbox:notRepairedDamage, data = df_cont)

#interaction terms (0.6597)
chosen_model=lm(formula = price ~ powerPS + kilometer + brand + fuelType + 
                  vehicleType + age + notRepairedDamage + gearbox + powerPS:fuelType + 
                  age:kilometer + gearbox:fuelType + brand:notRepairedDamage + vehicleType:notRepairedDamage + 
                  gearbox:kilometer + gearbox:age +brand:vehicleType + brand:gearbox + 
                  brand:gearbox:notRepairedDamage + brand:kilometer:vehicleType:notRepairedDamage + I(age^2):vehicleType, data = df_cont)



##logistic regression
#base model
model_binary=glm(formula=boolean ~ ., family = "binomial", data = df_train, control = list(maxit = 50))
new_data=data.frame(df_val)
predict_binary=predict(model_binary, new_data, type="response")
predict_binary_boolean=ifelse(predict_binary< 0.5, 0, 1)
pred=prediction(predict_binary_boolean,df_val$boolean)
perf_binary=performance(pred,"tpr","fpr")
auc=performance(pred,"auc")
plot(perf_binary)

#calibration plot base model
#round calculated probabilities to nearest 5% (e.g. 6% -> 5%)
predict_binary_rounded=round(predict_binary*2, 1) / 2
pred_vals=matrix(predict_binary_rounded,nrow=length(predict_binary_rounded))
empirical_vals=matrix(df_val$boolean, nrow=length(df_val$boolean))
combined=data.frame(pred_vals, empirical_vals)
names(combined)=c('predicted', 'empirical')

calibration_base = combined %>% 
  group_by(predicted) %>% 
  summarise(hit=mean(empirical), size=n(), probability = mean(predicted))

p=ggplot(calibration_base, aes(x=probability, y=hit))+
  geom_point(aes(size = size))+
  geom_abline(intercept=0, slope=1, linetype="dotted")+
  labs(x="Hit Rate Model", y="Hit Rate Empirical")+
  ggtitle("Calibration Graph [Base]") + 
  ggsave(file="mse231_project_cal1.pdf", width=8, height=5)


#test model
#model_1 (0.515)
#test_binary_1=glm(formula=boolean ~ price + brand + vehicleType + kilometer + I(kilometer^2), 
#                  family = "binomial", data = df_train, control = list(maxit = 50))

#model_2 (0.52) < - FAVORITE SO FAR
#test_binary_1=glm(formula=boolean ~ price + brand + vehicleType + kilometer + I(kilometer^2) + I(price^2), 
#                  family = "binomial", data = df_train, control = list(maxit = 50))

#model_3 (0.522)
#test_binary_1=glm(formula=boolean ~ price + brand + vehicleType + kilometer + I(kilometer^2) + I(price^2) + I(kilometer^3) + I(price^3), 
#family = "binomial", data = df_train, control = list(maxit = 50))


#Current Final Model
test_binary_1=glm(formula=boolean ~ price + brand + vehicleType + 
                    kilometer + I(kilometer^2) + I(price^2) + I(kilometer^3) +
                    I(price^3) + notRepairedDamage:brand + powerPS:fuelType + 
                    brand:vehicleType:gearbox, 
                  family = "binomial", data = df_train, control = list(maxit = 50))
predict_test_1=predict(test_binary_1, new_data, type="response")
predict_test_1_boolean=ifelse(predict_test_1< 0.5, 0, 1)
pred_test_1=prediction(predict_test_1_boolean,df_val$boolean)
perf_test_1=performance(pred_test_1,"tpr","fpr")
auc_test_1=performance(pred_test_1,"auc")
plot(perf_test_1)

#calibration plot
predict_test_1_rounded=round(predict_test_1*2, 1) / 2
pred_vals_1=matrix(predict_test_1_rounded,nrow=length(predict_test_1_rounded))
empirical_vals=matrix(df_val$boolean, nrow=length(df_val$boolean))
combined=data.frame(pred_vals_1, empirical_vals)
names(combined)=c('predicted', 'empirical')

calibration_base = combined %>% 
  group_by(predicted) %>% 
  summarise(hit=mean(empirical), size=n(), probability = mean(predicted))

p2=ggplot(calibration_base, aes(x=probability, y=hit))+
  geom_point(aes(size = size))+
  geom_abline(intercept=0, slope=1, linetype="dotted")+
  labs(x="Hit Rate Model", y="Hit Rate Empirical")+
  ggtitle("Calibration Graph [Final Model]") + 
  ggsave(file="mse231_project_cal2.pdf", width=8, height=5)

#print Area Under the Curve values
auc
auc_test_1
