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

file_path='/kyukon/data/gent/vo/000/gvo00041/vsc46127/validation_data_FINAL.csv'
val = pd.read_csv(file_path)

file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/subsample_2.csv'
train = pd.read_csv(file_path)



train = train.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})
val = val.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})
val['T2M_difference']=val['T_TARGET'] - val['T2M']
val['T2M']=val['T2M']-273.15
val['SKT']=val['SKT']-273.15
train['T2M_difference']=train['T_TARGET'] - train['T2M']
train['T2M']=train['T2M']-273.15
train['SKT']=train['SKT']-273.15

train = train.dropna(subset=['T2M_difference'])
val = val.dropna(subset=['T2M_difference'])

# Count NaN values per column
val['CBH'] = val['CBH'].fillna(0)
# Drop all rows with NaN values in any column
val = val.dropna()
train['CBH'] = train['CBH'].fillna(0)
# Drop all rows with NaN values in any column
train = train.dropna()



X_val = val[['LC_CORINE','LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF','NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
X_train = train[['LC_CORINE','LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF','NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
y_val=val['T2M_difference']
y_train=train['T2M_difference']




print("NaN values in X_train:", X_train.isnull().sum().sum())
print("NaN values in X_val:", X_val.isnull().sum().sum())
print("NaN values in y_train:", y_train.isnull().sum())
print("NaN values in y_val:", y_val.isnull().sum())

print("Infinity values in X_train:", np.isinf(X_train).sum().sum())
print("Infinity values in X_val:", np.isinf(X_val).sum().sum())
print("Infinity values in y_train:", np.isinf(y_train).sum())
print("Infinity values in y_val:", np.isinf(y_val).sum())

print("Values too large in X_train:", (X_train > np.finfo(np.float32).max).sum().sum())
print("Values too large in X_val:", (X_val > np.finfo(np.float32).max).sum().sum())
print("Values too large in y_train:", (y_train > np.finfo(np.float32).max).sum())
print("Values too large in y_val:", (y_val > np.finfo(np.float32).max).sum())














# Assuming you have your data in X_train, y_train, X_validation, and X_test
default_max_depth = 13      #increase perforance when increasing this, but also more computing time: good optimimum
default_n_estimators = 35   #bcs based on previous small models: no clear significance
default_max_features = 0.4    #1/3 variables


# Define the lists of parameter values to search
list_max_depth = [4, 8, 12, 16, 20, 24]
list_n_estimators = [10, 20, 30, 40, 50, 60]
list_max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]



# Initialize lists to store results
rmse_results_train = []
rmse_results_val = []
elapsed_times = []

for max_features_i in list_max_features:
    start_time = time.perf_counter()

    # Create a RandomForestRegressor with the current parameters
    rf = RandomForestRegressor(
        max_depth=default_max_depth,
        n_estimators=default_n_estimators,
        max_features=max_features_i,
        n_jobs=-1,
        random_state=42
    )

    # Fit the model on the training data
    rf.fit(X_train, y_train)
    y_pred_train = rf.predict(X_train)

    # Make predictions on the val set
    y_pred_val = rf.predict(X_val)

    # Evaluate the model on the datasets
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

    # Store results
    rmse_results_train.append(rmse_train)
    rmse_results_val.append(rmse_val)

    # Store elapsed time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    elapsed_times.append(elapsed_time)

    # Write results to file
    with open('run_time.txt', 'a') as f:
        f.write(f"Elapsed time model max_features: {max_features_i}: {elapsed_time} seconds and val score of {rmse_val} and train score of {rmse_train} \n")

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'max_features': list_max_features,
    'rmse_train': rmse_results_train,
    'rmse_val': rmse_results_val,
    'elapsed_time': elapsed_times
})


# Save DataFrame to CSV
results_df.to_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_features_2.csv', index=False)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(list_max_features, rmse_results_train, label='Train Set')
plt.plot(list_max_features, rmse_results_val, label='Validation Set')
plt.title('Random Forest Hyperparameter Tuning')
plt.xlabel('max_features')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('/kyukon/data/gent/vo/000/gvo00041/vsc46127/influence_max_features2.png')
plt.show()
plt.close()



# Initialize lists to store results
rmse_results_train = []
rmse_results_val = []
elapsed_times = []

for n_estimators_i in list_n_estimators:
    start_time = time.perf_counter()
    with open('run_time.txt', 'a') as f:
        f.write(f"started n_estimators: {n_estimators_i}\n")

    # Create a RandomForestRegressor with the current parameters
    rf = RandomForestRegressor(
        max_depth=default_max_depth,
        n_estimators=n_estimators_i,
        max_features=default_max_features,
        n_jobs=-1,
        random_state=42
    )

    # Fit the model on the training data
    rf.fit(X_train, y_train)
    y_pred_train = rf.predict(X_train)

    # Make predictions on the val set
    y_pred_val = rf.predict(X_val)

    # Evaluate the model on the datasets
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

    # Store results
    rmse_results_train.append(rmse_train)
    rmse_results_val.append(rmse_val)

    # Store elapsed time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    elapsed_times.append(elapsed_time)

    # Write results to file
    with open('run_time.txt', 'a') as f:
        f.write(f"Elapsed time model n_estimators: {n_estimators_i}: {elapsed_time} seconds and val score of {rmse_val} and train score of {rmse_train} \n")

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'n_estimators': list_n_estimators,
    'rmse_train': rmse_results_train,
    'rmse_val': rmse_results_val,
    'elapsed_time': elapsed_times
})


# Save DataFrame to CSV
results_df.to_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_n_estimators_2.csv', index=False)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(list_n_estimators, rmse_results_train, label='Train Set')
plt.plot(list_n_estimators, rmse_results_val, label='Validation Set')
plt.title('Random Forest Hyperparameter Tuning')
plt.xlabel('n_estimators')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('/kyukon/data/gent/vo/000/gvo00041/vsc46127/influence_n_estimators2.png')
plt.show()
plt.close()






# Initialize lists to store results
rmse_results_train = []
rmse_results_val = []
elapsed_times = []

for max_depth_i in list_max_depth:
    start_time = time.perf_counter()

    # Create a RandomForestRegressor with the current parameters
    rf = RandomForestRegressor(
        max_depth=max_depth_i,
        n_estimators=default_n_estimators,
        max_features=default_max_features,
        n_jobs=-1,
        random_state=42
    )

    # Fit the model on the training data
    rf.fit(X_train, y_train)
    y_pred_train = rf.predict(X_train)

    # Make predictions on the val set
    y_pred_val = rf.predict(X_val)

    # Evaluate the model on the datasets
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))

    # Store results
    rmse_results_train.append(rmse_train)
    rmse_results_val.append(rmse_val)

    # Store elapsed time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    elapsed_times.append(elapsed_time)

    # Write results to file
    with open('run_time.txt', 'a') as f:
        f.write(f"Elapsed time model max_depth: {max_depth_i}: {elapsed_time} seconds and val score of {rmse_val} and train score of {rmse_train} \n")

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'max_depth': list_max_depth,
    'rmse_train': rmse_results_train,
    'rmse_val': rmse_results_val,
    'elapsed_time': elapsed_times
})


# Save DataFrame to CSV
results_df.to_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_depth_2.csv', index=False)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(list_max_depth, rmse_results_train, label='Train Set')
plt.plot(list_max_depth, rmse_results_val, label='Validation Set')
plt.title('Random Forest Hyperparameter Tuning')
plt.xlabel('max_depth')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('/kyukon/data/gent/vo/000/gvo00041/vsc46127/influence_max_depth2.png')
plt.show()
plt.close()






