#import xarray as xr
#import rioxarray as rxr
import os
import numpy as np  
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
import plotly.express as px
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error



# Define functions for MAE, MSE, and correlation (R-squared)
def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def calculate_correlation(y_true, y_pred):
    return r2_score(y_true, y_pred)

def calculate_bias(y_pred, y_true):
    return ((y_pred - y_true).mean())




file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/validation_data_FINAL.csv'
val = pd.read_csv(file_path, usecols=[ 'city','T_TARGET','T_2M_COR', 'LC_CORINE', 'IMPERV', 'HEIGHT',  'COAST', 'ELEV', 'POP','NDVI', 'T_SK', 'SM', 'RH', 'SP', 'PRECIP', 'WS', 'WD', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR','SOLAR_ELEV', 'DECL'])

file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER2_2cities.csv'
train = pd.read_csv(file_path, usecols=['city','T_TARGET','T_2M_COR','LC_CORINE',  'IMPERV', 'HEIGHT',  'COAST', 'ELEV', 'POP', 'NDVI', 'T_SK', 'SM', 'RH', 'SP', 'PRECIP', 'WS', 'WD', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR','SOLAR_ELEV', 'DECL'])

# Count NaN values per column before dropping
val_nan_before_drop = val.isnull().sum()
train_nan_before_drop = train.isnull().sum()

# Drop rows with NaN values in 'T_TARGET' column
val = val.dropna(subset=['T_TARGET'])
train = train.dropna(subset=['T_TARGET'])

train = train.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})
val = val.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})

val['T2M_difference'] = val['T_TARGET'] - val['T2M']
val['T2M'] = val['T2M'] - 273.15
val['SKT'] = val['SKT'] - 273.15
train['T2M_difference'] = train['T_TARGET'] - train['T2M']
train['T2M'] = train['T2M'] - 273.15
train['SKT'] = train['SKT'] - 273.15

train = train.dropna(subset=['T2M_difference'])
val = val.dropna(subset=['T2M_difference'])

# Count NaN values per column after dropping 'T_TARGET'
val_nan_after_drop = val.isnull().sum()
train_nan_after_drop = train.isnull().sum()

# Fill NaN values with 0 in 'CBH' column
val['CBH'] = val['CBH'].fillna(0)
train['CBH'] = train['CBH'].fillna(0)

# Count NaN values per column after filling with 0
val_nan_after_fill = val.isnull().sum()
train_nan_after_fill = train.isnull().sum()

start_time = time.perf_counter()
# Fill NaN values with the median value of the specific 'City'
for column in val.columns:
    if val[column].isnull().any():
        val[column] = val.groupby('city')[column].transform(lambda x: x.fillna(x.median()))

for column in train.columns:
    if train[column].isnull().any():
        train[column] = train.groupby('city')[column].transform(lambda x: x.fillna(x.median()))

end_time = time.perf_counter()
elapsed_time = end_time - start_time

# Write results to file
with open('run_time.txt', 'a') as f:
    f.write(f"Elapsed time impute:  {elapsed_time} seconds  \n")




# Count NaN values per column after filling with median
val_nan_after_median_fill = val.isnull().sum()
train_nan_after_median_fill = train.isnull().sum()

# Calculate number of NaN values filled and dropped per column
val_nan_filled = val_nan_before_drop - val_nan_after_fill
train_nan_filled = train_nan_before_drop - train_nan_after_fill

val_nan_dropped = val_nan_after_drop - val_nan_after_median_fill
train_nan_dropped = train_nan_after_drop - train_nan_after_median_fill

# Writing results to a file
with open('run_time.txt', 'a') as f:
    f.write("NaN values filled per column before dropping 'T_TARGET':\n")
    f.write("Validation Data:\n")
    f.write(val_nan_filled.to_string() + "\n")
    f.write("Training Data:\n")
    f.write(train_nan_filled.to_string() + "\n")
    f.write("\n")
    f.write("NaN values dropped per column after dropping 'T_TARGET':\n")
    f.write("Validation Data:\n")
    f.write(val_nan_dropped.to_string() + "\n")
    f.write("Training Data:\n")
    f.write(train_nan_dropped.to_string() + "\n")

y_train = train['T2M_difference']
train = train[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC',  'CAPE', 'BLH', 'SSR','SOLAR_ELEV', 'DECL']]

y_val = val['T2M_difference'] 
val = val[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC',  'CAPE', 'BLH', 'SSR','SOLAR_ELEV', 'DECL']]




max_depth = 12     
n_estimators = 20  
max_features = 0.33   


# Create a RandomForestRegressor with the current parameters
rf = RandomForestRegressor(
     max_depth=max_depth,
     n_estimators=n_estimators,
     max_features=max_features,
     n_jobs=-1,
     random_state=42
   )

start_time = time.perf_counter()

with joblib.parallel_backend(backend='loky', n_jobs=-1):
    rf.fit(train, y_train)

# Fit the model on the training data
y_pred_train = rf.predict(train)

# Make predictions on the test set

# Make predictions on the validation set
y_pred_val = rf.predict(val)

# Evaluate the model on the datasets
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

mae_train= calculate_mae(y_train, y_pred_train)
mae_val= calculate_mae(y_val, y_pred_val)


mse_train= calculate_mse(y_train, y_pred_train)
mse_val= calculate_mse(y_val, y_pred_val)


correlation_train=  calculate_correlation(y_train, y_pred_train)
correlation_val= calculate_correlation(y_val, y_pred_val)

bias_train= calculate_bias(y_train, y_pred_train)
bias_val= calculate_bias(y_val, y_pred_val)




end_time = time.perf_counter()
elapsed_time = end_time - start_time
with open('run_time.txt', 'a') as f:
    f.write(f"Elapsed time model : {elapsed_time} seconds and rmse val score of {rmse_val} and rmse train score of {rmse_train} \n")
    f.write(f"Elapsed time model: {elapsed_time} seconds and mae val score of {mae_val} and mae train score of {mae_train} \n")
    f.write(f"Elapsed time model: {elapsed_time} seconds and mse val score of {mse_val} and mse train score of {mse_train} \n")
    f.write(f"Elapsed time model: {elapsed_time} seconds and bias val score of {bias_val} and bias train score of {bias_train} \n")
    f.write(f"Elapsed time model: {elapsed_time} seconds and correlation val score of {correlation_val} and correlation train score of {correlation_train} \n")


joblib.dump(rf, "/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/model_2cities_CL2.joblib")


