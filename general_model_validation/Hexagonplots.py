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
import seaborn as sns
import os
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import codecs
celcius='\u00B0C'





file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/test_data_FINAL.csv'
test = pd.read_csv(file_path, usecols=['ruralurbanmask','LCZ','T_2M', 'city','T_TARGET','T_2M_COR', 'LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL'])

# Count NaN values per column before dropping
test_nan_before_drop = test.isnull().sum()

# Drop rows with NaN values in 'T_TARGET' column
test = test.dropna(subset=['T_TARGET'])

test = test.rename(columns={'T_2M': 'T2M_NC'})
test = test.rename(columns={'T_2M_COR': 'T2M'})

test['T2M_difference'] = test['T_TARGET'] - test['T2M']
test['T2M'] = test['T2M'] - 273.15
test['T2M_NC'] = test['T2M_NC'] - 273.15
test['T_TARGET']=test['T_TARGET']-273.15

test = test.dropna(subset=['T2M_difference'])

# Count NaN values per column after dropping 'T_TARGET'
test_nan_after_drop = test.isnull().sum()

# Fill NaN values with 0 in 'CBH' column

# Count NaN values per column after filling with 0
test_nan_after_fill = test.isnull().sum()

start_time = time.perf_counter()
# Fill NaN values with the median value of the specific 'City'
for column in test.columns:
    if test[column].isnull().any():
        test[column] = test.groupby('city')[column].transform(lambda x: x.fillna(x.median()))


end_time = time.perf_counter()
elapsed_time = end_time - start_time

# Write results to file
with open('run_time.txt', 'a') as f:
    f.write(f"Elapsed time impute:  {elapsed_time} seconds  \n")




# Count NaN values per column after filling with median
test_nan_after_median_fill = test.isnull().sum()

# Calculate number of NaN values filled and dropped per column
test_nan_filled = test_nan_before_drop - test_nan_after_fill

test_nan_dropped = test_nan_after_drop - test_nan_after_median_fill

# Writing results to a file
with open('run_time.txt', 'a') as f:
    f.write("NaN values filled per column before dropping 'T_TARGET':\n")
    f.write("Test Data:\n")
    f.write(test_nan_filled.to_string() + "\n")
    f.write("\n")
    f.write("NaN values dropped per column after dropping 'T_TARGET':\n")
    f.write("Test Data:\n")
    f.write(test_nan_dropped.to_string() + "\n")


y_test = test['T2M_difference'] 
X_test = test[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]



# Load the trained model
model = joblib.load("/kyukon/data/gent/vo/000/gvo00041/vsc46127/FEATURESEL/model_FINAL.joblib")

# Predict
y_pred = model.predict(X_test)
test['y_pred'] = y_pred + test['T2M']
test['y_test'] =test['T_TARGET']
test['ypred_target']=y_pred
test['ytest_target']=y_test





import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame city_data with columns 'T2M', 'y_pred', 'y_test', 'ypred_target', and 'ytest_target'
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


import matplotlib.pyplot as plt

# Density scatter plot for T2M vs y_pred
plt.figure(figsize=(8, 6))
hb = plt.hexbin(y=test['y_pred'], x=test['y_test'], gridsize=50, cmap='inferno', bins='log', mincnt=1)
plt.plot([test['y_test'].min(), test['y_test'].max()], [test['y_test'].min(), test['y_test'].max()], color='red', linestyle='--')  # 1:1 line
cb = plt.colorbar(hb, label='counts')
cb.ax.tick_params(labelsize=14)  # Adjust colorbar tick label size
cb.set_label('Counts', fontsize=14)  # Adjust colorbar title font size
plt.title('Hexagonal binned plot of the temperature', fontsize=16)
plt.xlabel(f'UrbClim temperature [{celcius}]', fontsize=14)
plt.ylabel(f'RFmodel temperature [{celcius}]', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.savefig('y_pred_vs_y_test_density_scatter.png')  # Save the plot as a PNG image
plt.show()

# Density scatter plot for ypred_target vs ytest_target
plt.figure(figsize=(8, 6))
hb = plt.hexbin(y=test['ypred_target'], x=test['ytest_target'], gridsize=50, cmap='inferno', bins='log', mincnt=1)
plt.plot([test['ytest_target'].min(), test['ytest_target'].max()], [test['ytest_target'].min(), test['ytest_target'].max()], color='red', linestyle='--')  # 1:1 line
cb = plt.colorbar(hb, label='counts')
cb.ax.tick_params(labelsize=14)  # Adjust colorbar tick label size
cb.set_label('Counts', fontsize=14)  # Adjust colorbar title font size
plt.title('Hexagonal binned plot of the target variable', fontsize=16)
plt.xlabel(f'UrbClim temperature residual [{celcius}]', fontsize=14)
plt.ylabel(f'RFmodel temperature residual [{celcius}]', fontsize=14)
plt.tick_params(axis='both', labelsize=12)
plt.savefig('ypred_target_vs_ytest_target_density_scatter.png')  # Save the plot as a PNG image
plt.show()

