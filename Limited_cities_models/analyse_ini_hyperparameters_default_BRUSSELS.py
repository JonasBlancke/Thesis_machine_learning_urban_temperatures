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
import pandas as pd



file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/validation_data_BRUSSELS.csv'
val = pd.read_csv(file_path)

file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_data_BRUSSELS.csv'
train = pd.read_csv(file_path)


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




X_val = val[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
X_train = train[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
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
default_max_depth = 12      #increase perforance when increasing this, but also more computing time: good optimimum
default_n_estimators = 20   #bcs based on previous small models: no clear significance
default_max_features = 0.33   #1/3 variables


# Define the lists of parameter values to search
list_max_depth = [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]



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
        f.write(f"Elapsed time model GHENT, max depth: {max_depth_i}: {elapsed_time} seconds and val score of {rmse_val} and train score of {rmse_train} \n")

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'max_depth': list_max_depth,
    'rmse_train': rmse_results_train,
    'rmse_val': rmse_results_val,
    'elapsed_time': elapsed_times
})


# Save DataFrame to CSV
results_df.to_csv('/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_depth_BRUSSELS.csv', index=False)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(list_max_depth, rmse_results_train, label='Train Set')
plt.plot(list_max_depth, rmse_results_val, label='Validation Set')
plt.title('Random Forest Hyperparameter Tuning', fontsize=16)
plt.xlabel('max_depth', fontsize=16)
plt.ylabel('RMSE', fontsize=16)
plt.legend(fontsize=16)
plt.savefig('/kyukon/data/gent/vo/000/gvo00041/vsc46127/influence_max_depth_BRUSSELS.png')
plt.show()
plt.close()





