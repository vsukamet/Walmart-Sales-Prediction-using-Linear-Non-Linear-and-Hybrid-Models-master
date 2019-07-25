from statsmodels.tsa.seasonal import seasonal_decompose
import xgboost as xgb
import itertools
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import matplotlib
matplotlib.rc('xtick', labelsize=40)
matplotlib.rc('ytick', labelsize=40)
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import statsmodels.api as sm

#Reading the Preprocessed Data
data=pd.read_csv('mydata.csv')
HybridArima=[]
HybridXgBoost=[]

for i in data.Type.unique():
    #Selecting the data For Each Type of Stores
    ts=data[data.Type==i]
    #Selecting all the Stores of Department-1
    ts=ts[ts.Dept==1]
    ts['Date']=pd.to_datetime(ts['Date'])
    tss=ts.copy()
    ts=ts.set_index('Date')

    #Tuning the Parameters of Arima to get Order and Seasonal attributes
    print('Results from Dickey Fuller Test:')
    dtest = adfuller(ts['Weekly_Sales'], autolag='AIC')
    output = pd.Series(dtest[0:4], index=['Test Statistic', 'p-value', '#lags Used', 'Number of Observations Used'])
    for k, val in dtest[4].items():
        output['Critical Value (%s)'%k] = val
    print(output)

    #Giving the Parameters of p,d and q iteratively
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    sea_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    good_aic = np.inf
    good_pdq = None
    good_sea_pdq = None
    t_model = None
    forecastsofarima = pd.Series()
    for j in ts.Store.unique():
        ts1=ts[ts['Store']==j]
        for param in pdq:
            for param_seasonal in sea_pdq:
                try:
                    warnings.filterwarnings("ignore")
                    t_model = sm.tsa.statespace.SARIMAX(ts1['Weekly_Sales'],
                                             order = param,
                                             seasonal_order = param_seasonal,
                                             enforce_stationarity=False,
                                             enforce_invertibility=False)
                    results = t_model.fit(disp=0)
                    if results.aic < good_aic:
                        good_aic = results.aic
                        good_pdq = param
                        good_sea_pdq = param_seasonal
                except:
                    continue
        good_model = sm.tsa.statespace.SARIMAX(ts1['Weekly_Sales'],
                                      order=good_pdq,
                                      seasonal_order=good_sea_pdq,
                                      enforce_stationarity=False,
                                      enforce_invertibility=False)


        #Fitting the Tuned parameters to Arima to get Forecasts
        good_results = good_model.fit(disp=0)
        pred_d = good_results.get_prediction(start=pd.to_datetime('2010-02-05'),full_results=True)
        pred_dynamic_ci = pred_d.conf_int()
        ts_forecasted = pred_d.predicted_mean
        forecastsofarima=forecastsofarima.append(ts_forecasted)
        ts_truth = ts1['2010-02-05':]

        ts_truth1=ts_truth['Weekly_Sales'].values
        sc1 = MinMaxScaler(feature_range=(0,4300))
        ts_truth1=ts_truth1.reshape(ts_truth1.shape[0],1)
        ts_truth1=sc1.fit_transform(ts_truth1[:,:])
        ts_forecasted1=ts_forecasted.values
        ts_forecasted1=ts_forecasted1.reshape(ts_forecasted1.shape[0],1)
        ts_forecasted1=sc1.fit_transform(ts_forecasted1[:,:])

        #Calculating the RMSE for Predictions of Arima
        rmse=np.sqrt(mean_squared_error(ts_forecasted1,ts_truth1))
        # rmse=np.sqrt(mean_squared_error(ts_forecasted,ts_truth['Weekly_Sales']))
        HybridArima.append(rmse)

        print('Arima Root Mean Squared Error of our Type{} and store {} forecasts is {}'.format(i,j,round(rmse, 4)))


     #Extracting the Residue From the Forecasts of Arima
    result = seasonal_decompose(forecastsofarima, model='Additive',freq=56)
    forecastsofarima=forecastsofarima.reset_index()
    final=result.resid
    final=final.reset_index()
    final=final.set_index("index")
    tss=tss.set_index('Date')
    #Replacing the Actual Weekly-Sales by Residue Leftover by Arima
    tss['Weekly_Sales']=final[0]
    hybrid=tss.fillna(0)
    hybrid=hybrid.reset_index()
    arima_predictions=forecastsofarima.values
    actual_predictions=ts.values
    #Sending the Residue(With Added Features) to XGBOOST ALGORITHM
    for k in hybrid.Store.unique():

        t1=hybrid[hybrid['Store']==k]
        t=t1.copy()
        t=t.set_index('Date')

        del t1['Date']
        start=t1.index[0]
        end=max(t1.index)+1
        # print(start)
        # print(end)
        ta=t1.values
        ta1=ta
        X, Y = t1.iloc[:,:-1],t1.iloc[:,-1]

        #Tuning the XGBOOST PARAMETERS
        xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.4,reg_alpha=0.005, learning_rate = 0.21,max_depth = 1, alpha = 10, n_estimators =1000,min_child_weight=10, gamma=0.2, nthread=4, scale_pos_weight=1)
        datamatrix=xgb.DMatrix(data=X,label=Y)
        xgpara=xg_reg.get_xgb_params()
        cv_results = xgb.cv(dtrain=datamatrix, params=xgpara, nfold=6,num_boost_round=xg_reg.get_params()['n_estimators'],early_stopping_rounds=30,metrics="rmse", as_pandas=True, seed=123)
        xg_reg.set_params(n_estimators=cv_results.shape[0])
        sc = MinMaxScaler(feature_range=(0,1200))
        ta = sc.fit_transform(ta[:,:])
        xg_reg.fit(ta[:,:-1],ta[:,-1])


        #Getting the Predictions For Non-Linear Residue
        xgpred = xg_reg.predict(ta[:,:-1])
        xg_reg.fit(ta1[:,:-1],ta1[:,-1])
        xgpred1 = xg_reg.predict(ta1[:,:-1])
        arimaPreds1=arima_predictions[start:end,1]
        arimaPreds2=arimaPreds1.reshape((arimaPreds1.shape[0],1))
        arimaPreds3=sc.fit_transform(arimaPreds2[:,:])
        xgpred=xgpred.reshape((xgpred.shape[0],1))
        xgpred1=xgpred1.reshape((xgpred1.shape[0],1))


        #Adding the Predictions of XGBoost(Non-Linear Forecast) to Predictions of Arima(Linear Forecasts) to get Hybrid Forecasts
        final_preds = arimaPreds3.__add__(xgpred)
        final_preds1 = arimaPreds2.__add__(xgpred1)
        actual_predictions1=actual_predictions[start:end,-1]
        actual_predictions1=actual_predictions1.reshape((actual_predictions1.shape[0],1))
        actual_predictions2=sc.fit_transform(actual_predictions1[:,:])

        #Calculating the RMSE for Actual Sales and Hybrid ForeCasts
        rmse = np.sqrt(mean_squared_error(actual_predictions2, final_preds))
        HybridXgBoost.append(rmse)
        print("RMSE of Type {} STORE {} for Hybrid Case {}".format(i,k,round(rmse,4)))


        #Different Plots Obtained for Different Stores
        plt.title('Predictions for Type {} Store {}'.format(i,k))
        plt.scatter(t.index,actual_predictions1)
        plt.scatter(t.index,arimaPreds1)
        plt.scatter(t.index,final_preds1)
        plt.plot(t.index,actual_predictions1,label='Actual')
        plt.plot(t.index,arimaPreds1,label='Arima Predictions')
        plt.plot(t.index,final_preds1, label='ARIMA+XGBoost(Hybrid) Predictions')
        plt.xlabel('Date')
        plt.ylabel('Weekly_Sales')
        plt.legend(loc='best')
        #to plot uncomment plt.show()
        # plt.show()
finalrmse1=np.mean(HybridArima)
finalrmse2=np.mean(HybridXgBoost)
print('RMSE for ARIMA',finalrmse1)
print('RMSE for Hybrid Case',finalrmse2)











