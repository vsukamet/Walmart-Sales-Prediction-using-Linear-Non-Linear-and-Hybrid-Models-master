#Implementation of Holt's, Holt's Winter, Neural Network and Hybrid (Neural Network + Holt's Winter)
#By Ravi Bhushan and Anirudh Kumar Mullapulli

import numpy as np
import pandas as pd
import dateutil
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing,Holt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

dataset = pd.read_csv("input/train.csv", names=['Store','Dept','Date','weeklySales','isHoliday'],sep=',', header=0)
features = pd.read_csv("input/features.csv",sep=',', header=0,
                       names=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4',
                              'MarkDown5','CPI','Unemployment','IsHoliday']).drop(columns=['IsHoliday'])
stores = pd.read_csv("input/stores.csv", names=['Store','Type','Size'],sep=',', header=0)
dataset = dataset.merge(stores, how='left').merge(features, how='left')
dataset = dataset.fillna(0)
dataset['Date'] = dataset['Date'].apply(dateutil.parser.parse, yearfirst=True)

datasetLinear = dataset.drop(columns=["isHoliday", "Store", "Dept","Size", "Temperature", "CPI", "Fuel_Price", 'Unemployment', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'])
datasetLinearA = datasetLinear[datasetLinear.Type == 'A']
datasetLinearA1 = datasetLinearA.drop(columns=["Type"])
datasetLinearA1 = datasetLinearA1.groupby('Date').mean()
meanOfASeries =  datasetLinearA1['weeklySales'].mean()

datasetNonLinearA = dataset[dataset.Type == 'A']
datasetNonLinearA['week'] = pd.to_numeric(datasetNonLinearA['Date'].dt.week)
datasetNonLinearA['year'] = pd.to_numeric(datasetNonLinearA['Date'].dt.year)
datasetNonLinearA1 = datasetNonLinearA.drop(columns=["Type", "Date"])
result = seasonal_decompose(datasetLinearA1, model='additive')
result.plot()
plt.show()
trend = result.trend.fillna(0)
seasonal = result.seasonal.fillna(0)
residual = result.resid.fillna(0)



X_train = datasetLinearA1.head(103)
X_test = datasetLinearA1.tail(40)
y_hat_avg = X_test.copy()

fit1 = Holt(np.asarray(X_train['weeklySales'])).fit(smoothing_level = 0.3,smoothing_slope = 0.1)
y_hat_avg['Holt_linear'] = fit1.forecast(len(X_test))

plt.subplot(2, 2, 1)
plt.plot(X_train['weeklySales'], label='Train')
plt.plot(X_test['weeklySales'], label='Test')
plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
plt.legend(loc='best')

rms = sqrt(mean_squared_error(X_test.weeklySales, y_hat_avg.Holt_linear))
print(rms)

y_hat_avg = X_test.copy()
fit1 = ExponentialSmoothing(np.asarray(X_train['weeklySales']) ,seasonal_periods=5 ,trend='add', seasonal='add',).fit()
y_hat_avg['Holt_Winter'] = fit1.forecast(len(X_test))

plt.subplot(2, 2, 2)
plt.plot(X_train['weeklySales'], label='Train')
plt.plot(X_test['weeklySales'], label='Test')
plt.plot(y_hat_avg['Holt_Winter'], label='Holt_winter')
plt.legend(loc='best')

rms = sqrt(mean_squared_error(X_test.weeklySales, y_hat_avg.Holt_Winter))
print(rms)


kf = KFold(n_splits=5)
splited = []

for name, group in datasetNonLinearA1.groupby(["Store", "Dept"]):
    group = group.reset_index(drop=True)
    trains_x = []
    trains_y = []
    tests_x = []
    tests_y = []
    if group.shape[0] <= 5:
        f = np.array(range(5))
        np.random.shuffle(f)
        group['fold'] = f[:group.shape[0]]
        continue
    fold = 0
    for train_index, test_index in kf.split(group):
        group.loc[test_index, 'fold'] = fold
        fold += 1
    splited.append(group)

splited = pd.concat(splited).reset_index(drop=True)
maxSales = splited['weeklySales'].max()
minSales = splited['weeklySales'].min()
# create scaler
scaler = MinMaxScaler()
# fit scaler on data
scaler.fit(splited)
# apply transform
normalized = scaler.transform(splited)
rms = 0
fold = 0
#for fold in range(5):
dataset_train = splited.loc[splited['fold'] != fold]
dataset_test = splited.loc[splited['fold'] == fold]
train_y = dataset_train['weeklySales']
train_x = dataset_train.drop(columns=['weeklySales', 'fold'])
test_y = dataset_test['weeklySales']
test_x = dataset_test.drop(columns=['weeklySales', 'fold'])
nn = MLPRegressor(hidden_layer_sizes=(300,),  activation='relu', verbose=True)
nn.fit(train_x, train_y)
predicted = pd.Series(nn.predict(test_x))
predictedData = pd.DataFrame({'predicted':predicted})
predictedData['predicted'] = predictedData['predicted']*(maxSales - minSales) + minSales
predictedData['actual'] = test_y*(maxSales - minSales) + minSales
predictedData.to_csv('predictedDataMLP.csv')
predictedData['predicted'].fillna(0)
predictedData['actual'].fillna(0)
rms_error = sqrt(mean_squared_error(test_y, predicted))
print(rms_error)

plt.subplot(2, 2, 3)
plt.plot(predictedData['actual'], label='Test')
plt.plot(predictedData['predicted'], label='NN')
plt.legend(loc='best')

trend = result.trend.fillna(meanOfASeries)
seasonal = result.seasonal.fillna(0)
residual = result.resid.fillna(0)
observed = result.observed.fillna(0)

datasetLinearA1H = datasetLinearA1
datasetLinearA1H['weeklySales'] = seasonal['weeklySales'] + meanOfASeries + (meanOfASeries - trend['weeklySales'])


X_train1 = datasetLinearA1H.head(103)
X_test1 = datasetLinearA1H.tail(40)

y_hat_avg = X_test1.copy()
fit1 = ExponentialSmoothing(np.asarray(X_train1['weeklySales']) ,seasonal_periods=5 ,trend='add', seasonal='add',).fit()
y_hat_avg['weeklySales'] = fit1.forecast(len(X_test1))
y_hat_avg['weeklySales'] = pd.DataFrame({'weeklySales':y_hat_avg['weeklySales']})
datasetNonLinearA = dataset[dataset.Type == 'A']
datasetNonLinearA1 = datasetNonLinearA.drop(columns=["Type"])
datasetNonLinearA1 = datasetNonLinearA1.groupby('Date').mean()
datasetNonLinearA1 = datasetNonLinearA1.drop(columns=['weeklySales'])
datasetNonLinearA1['weeklySales'] = residual['weeklySales'].fillna(0)
X_train = datasetNonLinearA1.head(103).fillna(0)
X_test = datasetNonLinearA1.tail(40).fillna(0)
train_y = X_train['weeklySales']
train_x = X_train.drop(columns=['weeklySales'])
test_y = X_test['weeklySales']
test_x = X_test.drop(columns=['weeklySales'])


nn = MLPRegressor(hidden_layer_sizes=(200,),  activation='relu', verbose=3)
nn.fit(train_x, train_y)
predicted = pd.Series(nn.predict(test_x))
predictedData = pd.DataFrame({'weeklySales_nn':predicted})
predictedData['weeklySales_holtz'] = y_hat_avg['weeklySales']

predictedData.to_csv('predictedData_hybrid.csv')
y_hat_avg['weeklySales'] = [x + y for x, y in zip(y_hat_avg['weeklySales'], predictedData['weeklySales_nn'])]
plt.subplot(2, 2, 4)
plt.plot(X_train1['weeklySales'], label='Train')
plt.plot(X_test1['weeklySales'], label='Test')
plt.plot(y_hat_avg['weeklySales'], label='Hybrid')
plt.legend(loc='best')
plt.show()

rms = sqrt(mean_squared_error(X_test1.weeklySales, y_hat_avg.weeklySales))
print(rms)

