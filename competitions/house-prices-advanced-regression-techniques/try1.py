#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:07:00 2018

@author: midiago
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
data_complete = pd.read_csv('train.csv')
#dataresult = pd.read_csv('sample_submission.csv')
#
#data_complete = pd.merge(dataset, dataresult, on='Id')



#data_complete.plot(kind='scatter',  x='MSSubClass', y='SalePrice')
#data_complete.plot(kind='scatter',  x='LotFrontage', y='SalePrice')
#data_complete.plot(kind='scatter',  x='LotArea', y='SalePrice')
#data_complete.plot(kind='scatter',  x='OverallQual', y='SalePrice')
#
#data_complete.plot(kind='scatter',  x='OverallCond', y='SalePrice')
#data_complete.plot(kind='scatter',  x='YearRemodAdd', y='SalePrice')
#data_complete.plot(kind='scatter',  x='MasVnrArea', y='SalePrice')


data_complete['Alley']=data_complete["Alley"].replace(np.nan, 'NA', regex=True)


cleanup_nums = {"LotConfig":    {"Inside": 0, "Corner": 1, "CulDSac": 2, "FR2":3, "FR3":4},
                "MSZoning":     {"RL": 0, "RM": 1, "FV": 2, "RH": 3, "C (all)": 4 },
                "Street":       {"Pave": 0, "Grvl": 1},
                "LotShape":     {"Reg": 0, "IR1": 1, "IR2": 2, "IR3": 3},
                "Alley":        {"Grvl": 0, "Pave": 1, "NA": 2},
                "LandContour":  {"Lvl": 0, "Bnk": 1, "HLS": 2, "Low": 3},
                "Utilities":    {"AllPub": 0, "NoSewr": 1, "NoSeWa": 2, "ELO": 3},
                "LandSlope":    {"Gtl":0, "Mod": 1, "Sev": 2},
                "Neighborhood": {"Blmngtn":0, 
                                 "Blueste": 1, 
                                 "BrDale": 2, 
                                 "BrkSide": 3, 
                                 "ClearCr": 4, 
                                 "CollgCr": 5, 
                                 "Crawfor": 6, 
                                 "Edwards": 7, 
                                 "Gilbert": 8, 
                                 "IDOTRR": 9, 
                                 "MeadowV": 10, 
                                 "Mitchel": 11, 
                                 "NAmes": 12, 
                                 "NoRidge": 13, 
                                 "NPkVill": 14, 
                                 "NridgHt": 15, 
                                 "NWAmes": 16, 
                                 "OldTown": 17, 
                                 "SWISU": 18, 
                                 "Sawyer": 19, 
                                 "SawyerW": 20, 
                                 "Somerst": 21, 
                                 "StoneBr": 22, 
                                 "Timber": 23, 
                                 "Veenker": 24},
                
                }

data_complete.replace(cleanup_nums, inplace=True)




variables = ['MSSubClass', 
             'LotFrontage', 
             'LotArea', 
             'OverallQual', 
             'OverallCond', 
             'YearRemodAdd', 
             'MasVnrArea',
             'LotConfig', 
             'MSZoning',
             'Street',
             'LotShape',
#             'Alley',
             'LandContour',
#             'Utilities',
#             'LandSlope',
#             'Neighborhood',
             'Neigh_0', 'Neigh_1', 'Neigh_2', 'Neigh_3', 'Neigh_4', 'Neigh_5', 'Neigh_6', 'Neigh_7', 'Neigh_8', 'Neigh_9',
             'Neigh_10', 'Neigh_11', 'Neigh_12', 'Neigh_13', 'Neigh_14', 'Neigh_15', 'Neigh_16', 'Neigh_17', 'Neigh_18', 'Neigh_19',
             'Neigh_20', 'Neigh_21', 'Neigh_22', 'Neigh_23', 'Neigh_24',
             'SalePrice']



data_complete=pd.get_dummies(data_complete, columns=["Neighborhood"], prefix=["Neigh"])



my_dataset_complete = data_complete[variables]
my_dataset_complete = my_dataset_complete.dropna()
#X = dataset.iloc[:, :].values
X = my_dataset_complete.iloc[:, :-1] .values
y = my_dataset_complete.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

## Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1, 1))



# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
#regressor = LinearRegression()
#regressor = SVR(kernel = 'linear')
regressor = RandomForestRegressor(n_estimators = 200, random_state = 0)

regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


from sklearn.metrics import mean_squared_error
metric = np.sqrt(mean_squared_error(np.log(y_test), np.log(sc_y.inverse_transform(y_pred))))
metric


#
## Importing the dataset
#dataset_test = pd.read_csv('test.csv')
#dataset_test.replace(cleanup_nums, inplace=True)
#data_complete=pd.get_dummies(data_complete, columns=["Neighborhood"], prefix=["Neigh"])
#
#
#
#data_test = dataset_test[variables[:-1]]
#X_test_final = sc_X.transform(data_test)
#y_pred_test_final = regressor.predict(X_test_final)