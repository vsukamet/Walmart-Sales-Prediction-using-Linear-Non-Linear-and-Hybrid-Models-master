import pandas as pd
import numpy as np
import warnings
from math import sqrt
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing
warnings.filterwarnings("ignore")
dataset=pd.read_csv('TAD1.csv',header=0)
plt.figure(figsize=(10,10))
rms_list=[]
for i in dataset.Store.unique():
    data1=dataset[dataset['Store']==i]
    data1['Date'] = pd.to_datetime(data1['Date'])
    train=data1[::]
    test=data1[::]
    train=train.set_index('Date')
    test=test.set_index('Date')
    y_hat_avg = test.copy()
    fit2 = SimpleExpSmoothing(np.asarray(train['Weekly_Sales'])).fit(smoothing_level=0.3,optimized=False)
    y_hat_avg['SES'] = fit2.forecast(len(test))
    train['Weekly_Sales'].plot()
    test['Weekly_Sales'].plot()
    y_hat_avg['SES'].plot()
    rms = sqrt(mean_squared_error(test.Weekly_Sales, y_hat_avg.SES))
    rms_list.append(rms)
    print(rms)
plt.legend(loc='best')
plt.show()
print("RMSE obtained by taking mean of all rmse of all stores is" , np.mean(rms_list))
