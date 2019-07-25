####################################################
####### importing all the required package #########
####################################################

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math


######################################################
##### Reading the csv file into pandas ##############
#####################################################
data = pd.read_csv('Final.csv')
data.head()

#### Data pre processing #############################

store_one =data.groupby(['Store','Date']).mean().loc[1]
store_one = store_one.reset_index()
X = store_one[['Date','IsHoliday_x','Fuel_Price','CPI']]
y = store_one['Weekly_Sales']
X.Date = pd.to_datetime(X.Date)
X['year'] = X.Date.apply(lambda x: x.year)
X['month'] = X.Date.apply(lambda x: x.month)
X['week'] = X.Date.apply(lambda x: x.week)
X.drop('Date',axis=1,inplace=True)
X.head()

############# Splitting the data set into test and train ##########
n = int(0.7*len(X))
X_train = X.iloc[:n,:]
X_test = X.iloc[100:,:]

y_train = y.iloc[:n]
y_test = y.iloc[100:]
# print(y_train.head(),"\n",y_test.head())

########### fitting the Linear regression model ################
lr = LinearRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
print("Weights of Linear regression:",lr.coef_)
print("The RMSE error of the Linear Regression",math.sqrt(mean_squared_error(pred, y_test)))

################## Plot of the Fitted straight line ###################
plt.plot(X_test['week'],pred, label='Predicted values')
plt.plot(X_test['week'],y_test, label='Actual values')
plt.xticks(rotation=45)
plt.legend()
plt.show()

############## Plot of predicted values and actual values vs the weeks ##############3

plt.plot(y_test, pred,'ro')
plt.plot(y_test, y_test,'b-')
plt.legend()
plt.grid()
plt.show()

##################### Fitting the Random Forest Regression ###################

model = RandomForestRegressor(n_estimators=100,n_jobs=-1).fit(X_train,y_train)
pred_rf = model.predict(X_test)
############ Getting the RMSE value #########################
print("The RMSE error for Random forest is:",math.sqrt(mean_squared_error(pred_rf, y_test)))

########### Plot of predicted value and actual values vs the weeks #############
plt.plot(X_test['week'],pred_rf, label='Predicted values')
plt.plot(X_test['week'],y_test, label='Actual values')
plt.xticks(rotation=45)
plt.legend()
plt.show()
