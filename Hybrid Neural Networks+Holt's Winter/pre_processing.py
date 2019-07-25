# Data Manipulation - Gives the Heat Map and the dependency of weekly sales on non linear parameters
import numpy as np
import pandas as pd
import dateutil
import datetime
import matplotlib.pyplot as plt

import seaborn as sns; sns.set(style="ticks", color_codes=True)

dataset = pd.read_csv("input/train.csv", names=['Store','Dept','Date','weeklySales','isHoliday'],sep=',', header=0)
features = pd.read_csv("input/features.csv",sep=',', header=0,
                       names=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4',
                              'MarkDown5','CPI','Unemployment','IsHoliday']).drop(columns=['IsHoliday'])
stores = pd.read_csv("input/stores.csv", names=['Store','Type','Size'],sep=',', header=0)
dataset = dataset.merge(stores, how='left').merge(features, how='left')
dataset = dataset.fillna(0)
dataset['Date'] = dataset['Date'].apply(dateutil.parser.parse, yearfirst=True)

sns.pairplot(dataset, vars=['weeklySales', 'Fuel_Price', 'Size', 'CPI', 'Dept', 'Temperature', 'Unemployment'])
sns.pairplot(dataset.fillna(0), vars=['weeklySales', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'])

plt.subplot(2, 3, 1)
plt.scatter(dataset['MarkDown1'] , dataset['weeklySales'])
plt.ylabel('weeklySales')
plt.xlabel('MarkDown1')

plt.subplot(2, 3, 2)
plt.scatter(dataset['MarkDown2'] , dataset['weeklySales'])
plt.ylabel('weeklySales')
plt.xlabel('MarkDown2')

plt.subplot(2, 3, 3)
plt.scatter(dataset['MarkDown3'] , dataset['weeklySales'])
plt.ylabel('weeklySales')
plt.xlabel('MarkDown3')

plt.subplot(2, 3, 4)
plt.scatter(dataset['MarkDown4'] , dataset['weeklySales'])
plt.ylabel('weeklySales')
plt.xlabel('MarkDown4')

#plt.subplot(2, 3, 5)
#plt.scatter(dataset['MarkDown5'] , dataset['weeklySales'])
#plt.ylabel('weeklySales')
#plt.xlabel('MarkDown5')

#plt.subplot(2, 3, 6)
#plt.scatter(dataset['isHoliday'] , dataset['weeklySales'])
#plt.ylabel('weeklySales')
#plt.xlabel('isHoliday')
#plt.show()

