# -*- coding: utf-8 -*-
"""SHAPLEY_ALL.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1e2g1TBUb9Cw6BDtgTa8ndZO1Wm2FG1W1
"""

#import xarray as xr
#import rioxarray as rxr
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import time
import signal


np.random.seed(42)

# Define a function to handle the timeout
def timeout_handler(signum, frame):
    raise TimeoutError("The operation took too long. Terminating.")

# Set the timeout duration (in seconds)
timeout_duration = 100

start_time = time.perf_counter()


file_path = '/content/drive/MyDrive/thesis/Shapley/TEST_subsample_CL1.csv'
val1 = pd.read_csv(file_path, usecols=['city','T_2M','T_TARGET','T_2M_COR', 'LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR','SOLAR_ELEV', 'DECL'])
file_path = '/content/drive/MyDrive/thesis/Shapley/TEST_subsample_CL2.csv'
val2 = pd.read_csv(file_path, usecols=['city','T_2M','T_TARGET','T_2M_COR', 'LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR','SOLAR_ELEV', 'DECL'])
file_path = '/content/drive/MyDrive/thesis/Shapley/TEST_subsample_CL3.csv'
val3 = pd.read_csv(file_path, usecols=['city','T_2M','T_TARGET','T_2M_COR', 'LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR','SOLAR_ELEV', 'DECL'])
val = pd.concat([val1, val2, val3], ignore_index=True)
#CHANGE THIS FOR CLUSTER-SPECIFIC SHAPLEY VALUES: PART2

# Count NaN values per column before dropping
val_nan_before_drop = val.isnull().sum()

# Drop rows with NaN values in 'T_TARGET' column
val = val.dropna(subset=['T_TARGET'])

val = val.rename(columns={'T_2M': 'T2M_NC', 'T_2M_COR': 'T2M'})
#val = val.rename(columns={'T_SK': 'SKT'})

val['T2M_difference'] = val['T_TARGET'] - val['T2M']
val['T2M'] = val['T2M'] - 273.15
#val['SKT'] = val['SKT'] - 273.15
val['T2M_NC'] = val['T2M_NC'] - 273.15

val = val.dropna(subset=['T2M_difference'])

# Count NaN values per column after dropping 'T_TARGET'
val_nan_after_drop = val.isnull().sum()

# Fill NaN values with 0 in 'CBH' column
#val['CBH'] = val['CBH'].fillna(0)

# Count NaN values per column after filling with 0
val_nan_after_fill = val.isnull().sum()

# Fill NaN values with the median value of the specific 'City'
for column in val.columns:
    if val[column].isnull().any():
        val[column] = val[column].transform(lambda x: x.fillna(x.median()))



# Count NaN values per column after filling with median
val_nan_after_median_fill = val.isnull().sum()

# Calculate number of NaN values filled and dropped per column
val_nan_filled = val_nan_before_drop - val_nan_after_fill

val_nan_dropped = val_nan_after_drop - val_nan_after_median_fill



y_val = val['T2M_difference']
X_val = val[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC',  'CAPE', 'BLH', 'SSR','SOLAR_ELEV', 'DECL']]






start_time = time.perf_counter()


#WHICH MODEL YOU WANT TO USE:
#model=joblib.load("/content/drive/MyDrive/thesis/Shapley/model_2cities_CL3.joblib")
#model=joblib.load("/content/drive/MyDrive/thesis/Shapley/GENERAL_6cities.joblib")
#model=joblib.load("/content/drive/MyDrive/thesis/Shapley/model_FINAL.joblib")
model=joblib.load('/content/drive/MyDrive/thesis/Shapley/model_ALLcities_CL3.joblib')

import time
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# Calculate feature importances using Gini impurity
feature_importances_gini = model.feature_importances_

# Calculate feature importances using Permutation Importance
# Here, we'll use 10 repetitions for stability and a random state for reproducibility
result = permutation_importance(model, X_val.iloc[:1000], model.predict(X_val.iloc[:1000]), n_repeats=10, random_state=42)
feature_importances_permutation = result.importances_mean

# Create visualizations for Gini and Permutation importance

# Gini Importance Plot
plt.figure(figsize=(8, 6))
plt.barh(X_val.columns, feature_importances_gini, color='skyblue')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (Gini Impurity)')
plt.gca().invert_yaxis()  # Arrange features with highest importance at the top
plt.tight_layout()
plt.savefig('gini_importance.png')
 # Close the plot figure
plt.show()

# Permutation Importance Plot
plt.figure(figsize=(8, 6))
plt.barh(X_val.columns, feature_importances_permutation, color='coral')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (Permutation)')
plt.gca().invert_yaxis()  # Arrange features with highest importance at the top
plt.tight_layout()
plt.show()
 # Close the plot figure

print("Feature importance plots generated and saved successfully!")

!pip install fasttreeshap

import shap
feat_names = list(X_val.columns)
random_data_interv = X_val.sample(n=500)
explainer=shap.TreeExplainer(model, data=random_data_interv, model_output='raw', feature_perturbation='interventional')
shap_values_with_expected_value = explainer(X_val.sample(n=100000),check_additivity=False)
#N_shap_values_with_expected_value = explainer(X_val[X_val['SOLAR_ELEV'] < 0].iloc[3000:])  # Explain predictions using the first 1000 rows of X_val
#D_shap_values_with_expected_value = explainer(X_val[X_val['SOLAR_ELEV'] > 0].iloc[3000:])  # Explain predictions using the first 1000 rows of X_val

plt.figure(figsize=(18, 12))

shap.plots.beeswarm(shap_values_with_expected_value, max_display=20)
#plt.savefig('/content/drive/MyDrive/thesis/Shapley/beeswarm_plot_CL3_GENERAL.png')  # Save the figure as beeswarm_plot.png
plt.show()

plt.figure(figsize=(18, 12))

# Plot the violin plot for the top features

shap.plots.violin(shap_values_with_expected_value,feature_names=feat_names, plot_type='layered_violin', max_display=20)
#plt.savefig('/content/drive/MyDrive/thesis/Shapley/violin_plot_CL3_GENERAL.png')  # Save the figure as beeswarm_plot.png
plt.show()