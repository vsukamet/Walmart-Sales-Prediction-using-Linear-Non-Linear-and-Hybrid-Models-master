import pandas as pd
from sklearn.preprocessing import LabelEncoder

#Loading the Original Datasets
dfeat=pd.read_csv('features.csv')
dtrain=pd.read_csv('train.csv')
dstores=pd.read_csv('stores.csv')

#Changing the Date from Object to Datetime Format
dfeat['Date'] = pd.to_datetime(dfeat['Date'], format="%Y-%m-%d")
dtrain['Date'] = pd.to_datetime(dtrain['Date'], format="%Y-%m-%d")
dtest['Date'] = pd.to_datetime(dtest['Date'], format="%Y-%m-%d")

#Performing the different Merge Operations to Combined the Different sets of Features
train = pd.merge(dtrain, dstores, how="left", on="Store")
train = pd.merge(train, dfeat, how = "inner", on=["Store","Date"])

#Dropping the Extra Feature
train = train.drop(["IsHoliday_y"], axis=1)

#Removing the Null-Values
final_train = train.fillna(0)

#Replacing the Negative Values with Zeros
final_train.loc[final_train['Weekly_Sales'] < 0.0,'Weekly_Sales'] = 0.0
final_train.loc[final_train['MarkDown2'] < 0.0,'MarkDown2'] = 0.0
final_train.loc[final_train['MarkDown3'] < 0.0,'MarkDown3'] = 0.0

#Changing the Boolean or Categorical Features into the Numeric Format
cat_fea = ['IsHoliday_x','Type']
for col in cat_fea:
    lbl = LabelEncoder()
    lbl.fit(final_train[col].values.astype('str'))
    final_train[col] = lbl.transform(final_train[col].values.astype('str'))

#Re-oredering into the Desired Format
final_train = final_train[['Store', 'Dept', 'Date', 'Unemployment', 'IsHoliday_x', 'Type', 'Size',
       'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3',
       'MarkDown4', 'MarkDown5', 'CPI', 'Weekly_Sales']]

#CONTAINING THE COMPLETE DATA
final_train.to_csv("COMPLETE DATA.csv", index=False)

#EXTRACTING the DATA BASED ON THE TYPE OF STORES
ts=final_train[final_train.Type==0]
ts.to_csv(" TYPE 0 DATA.csv", index=False)

#EXTRACTING the DATA BASED ON THE DEPARTMENT OF STORES
ts_1=ts[ts.Dept==1]
ts.to_csv("STORE1.csv", index=False)




