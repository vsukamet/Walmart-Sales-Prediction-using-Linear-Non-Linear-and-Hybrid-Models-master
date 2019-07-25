import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

#Loading the Data of Different types of Stores
data=pd.read_csv('mydata.csv')
# data=data.drop(['CPI'],axis=1)
RMSE=[]


#Iterating through all different Stores of Same Type
for i in data.Store.unique():

    #Taking the Data for each Individual Store
    ts=data[data['Store']==i]
    ts1=ts.copy()
    ts['Date'] = pd.to_datetime(ts['Date'])
    ts1['Date'] = pd.to_datetime(ts1['Date'])
    ts1=ts1.set_index('Date')
    X, Y = ts1.iloc[:,:-1],ts1.iloc[:,-1]

    #Tuning the XGboost Parameters by performing Cross Validation
    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.4, learning_rate = 0.1,max_depth = 4, alpha = 10, n_estimators =1000,min_child_weight=10,reg_alpha=0.005, gamma=0, nthread=4, scale_pos_weight=1)
    datamatrix=xgb.DMatrix(data=X,label=Y)
    xgpara=xg_reg.get_xgb_params()
    cv_results = xgb.cv(dtrain=datamatrix, params=xgpara, nfold=6,num_boost_round=xg_reg.get_params()['n_estimators'],early_stopping_rounds=30,metrics="rmse", as_pandas=True, seed=123)
    xg_reg.fit(X,Y)

    #Predicting the Sales of Different Stores
    preds = xg_reg.predict(X)
    rmse = np.sqrt(mean_squared_error(Y, preds))
    RMSE.append(rmse)
    print("RMSE for XGboost for store {} is {}".format(i,round(rmse,4)))

    #Plotting Decision Trees used in prediction step
    xgb.plot_tree(xg_reg,num_trees=1)
    plt.rcParams['figure.figsize'] = [50, 10]
    plt.title('DECISION TREE FOR STORE {}'.format(i))
    #to plot uncomment plt.show()
    # plt.show()
    sort=np.argsort(xg_reg.feature_importances_)[::-1]
    for index in sort:
        print([data.columns[index], xg_reg.feature_importances_[index]])

    #plotting Feature Importance Curve For each Store
    plot_importance(xg_reg, max_num_features = 15)
    plt.title('FEATURE IMPORTANCE FOR STORE {}'.format(i))
    #to plot uncomment plt.show()
    # plt.show()


    #XGBOOST predictions Plots for Different Stores
    plt.title(' XGBoost-Predictions for store {}:'.format(i))
    plt.scatter(ts1.index,Y)
    plt.scatter(ts1.index,preds)
    plt.plot(ts1.index,Y,label='Actual')
    plt.plot(ts1.index,preds, label='Predicted')
    plt.legend(loc='best')
    #to plot uncomment plt.show()
    # plt.show()

AVGRMSE=np.mean(RMSE)
print('AVG RMSE for all stores',AVGRMSE)








