import statsmodels.api as sm 
from statsmodels.graphics.api import qqplot
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('xtick', labelsize=10) 
matplotlib.rc('ytick', labelsize=10) 
import pandas as pd

####################################################################
########## Import CSV file as pandas dataframe #####################
####################################################################

data=pd.read_csv("datasetLinear.csv")
# print(data.head())
####################################################################
##### Data pre processing ##########################################
####################################################################

dataset = pd.read_csv("train.csv", names=['Store','Dept','Date','weeklySales','isHoliday'],sep=',', header=0)
# print(dataset.head())
features = pd.read_csv("features.csv",sep=',', header=0,
                       names=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4',
                              'MarkDown5','CPI','Unemployment','IsHoliday']).drop(columns=['IsHoliday'])
stores = pd.read_csv("stores.csv", names=['Store','Type','Size'],sep=',', header=0)
# print("Stores:",stores.head())
dataset = dataset.merge(stores, how='left').merge(features, how='left')
dataset = dataset.fillna(0)
dataset['Date'] = pd.to_datetime(data['Date'])
#print("linear data")
datasetLinear = dataset.drop(columns=["isHoliday", "Store", "Dept","Size", "Temperature", "CPI", "Fuel_Price", 'Unemployment', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'])
datasetLinearA = datasetLinear[datasetLinear.Type == 'A']
datasetLinearA1 = datasetLinearA.drop(columns=["Type"])
datasetLinearA1 = datasetLinearA1.groupby('Date').mean()
# meanOfASeries =  datasetLinearA1['weeklySales'].mean()t_1[['weeklySales']]
# print("datasetLinear \n",datasetLinear.head())
# print("datasetLinearA \n",datasetLinearA.head())
print("datasetLinearA1\n",datasetLinearA1.head())

####################################################################
######## Implementing the ARMA model ###############################
####################################################################

data_copy = datasetLinearA1
arma_model1 = sm.tsa.ARMA(data_copy, (1,1)).fit(disp=False)
print("The parameters with (1,1)",arma_model1.params)

arma_model2 = sm.tsa.ARMA(data_copy, (3,0)).fit(disp=False)
print("The parameters with (3,0)",arma_model2.params)
print("\nThe AIC value comparison for different paramaters of AR and MA")
print(arma_model1.hqic,"\t",arma_model2.hqic)
print(arma_model1.aic,"\t",arma_model2.aic)
print(arma_model1.bic,"\t",arma_model2.bic)

######################################################################
####### Checking the residual values #################################
######################################################################
resid_30 = sm.stats.durbin_watson(arma_model2.resid.values)
resid_20 = sm.stats.durbin_watson(arma_model1.resid.values)

predict_values = arma_model2.predict(start='2010-02-05',dynamic = False)
# print(predict_values)

#####################################################################
########## Plotting the graph and RMSE values #######################
#####################################################################
import math
import numpy as np
ts_truth = datasetLinearA1['weeklySales'].values
# print(ts_truth)

# Compute the mean square error
rmse = math.sqrt(np.sum((((predict_values - ts_truth) ** 2)).mean()))

print('The Root Mean Squared Error of our forecasts is {}'.format(round(rmse, 4)))


plt.plot(predict_values, label='Predicted values',color = 'r')
plt.plot(datasetLinearA1['weeklySales'], label='Actual values',color = 'b')
plt.xticks(rotation=45)
plt.legend()
plt.show()

