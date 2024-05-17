#import xarray as xr
#import rioxarray as rxr
import os
import numpy as np  
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error
import joblib
import time
import matplotlib.pyplot as plt
from scipy.stats import pearsonr  # Add this line to import pearsonr
from sklearn.inspection import permutation_importance



file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/validation_data_FINAL.csv'
val = pd.read_csv(file_path, usecols=['T_2M', 'city','T_TARGET','T_2M_COR', 'LC_CORINE','LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF','NDVI', 'T_SK', 'SM', 'RH', 'SP', 'PRECIP', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL'])

# Count NaN values per column before dropping
val_nan_before_drop = val.isnull().sum()

# Drop rows with NaN values in 'T_TARGET' column
val = val.dropna(subset=['T_TARGET'])

val = val.rename(columns={'T_2M': 'T2M_NC'})
val = val.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})

val['T2M_difference'] = val['T_TARGET'] - val['T2M']
val['T2M'] = val['T2M'] - 273.15
val['SKT'] = val['SKT'] - 273.15
val['T2M_NC'] = val['T2M_NC'] - 273.15

val = val.dropna(subset=['T2M_difference'])

# Count NaN values per column after dropping 'T_TARGET'
val_nan_after_drop = val.isnull().sum()

# Fill NaN values with 0 in 'CBH' column
val['CBH'] = val['CBH'].fillna(0)

# Count NaN values per column after filling with 0
val_nan_after_fill = val.isnull().sum()

start_time = time.perf_counter()
# Fill NaN values with the median value of the specific 'City'
for column in val.columns:
    if val[column].isnull().any():
        val[column] = val.groupby('city')[column].transform(lambda x: x.fillna(x.median()))


end_time = time.perf_counter()
elapsed_time = end_time - start_time

# Write results to file
with open('run_time.txt', 'a') as f:
    f.write(f"Elapsed time impute:  {elapsed_time} seconds  \n")



# Count NaN values per column after filling with median
val_nan_after_median_fill = val.isnull().sum()

# Calculate number of NaN values filled and dropped per column
val_nan_filled = val_nan_before_drop - val_nan_after_fill

val_nan_dropped = val_nan_after_drop - val_nan_after_median_fill

# Writing results to a file
with open('run_time.txt', 'a') as f:
    f.write("NaN values filled per column before dropping 'T_TARGET':\n")
    f.write("Validation Data:\n")
    f.write(val_nan_filled.to_string() + "\n")
    f.write("\n")
    f.write("NaN values dropped per column after dropping 'T_TARGET':\n")
    f.write("Validation Data:\n")
    f.write(val_nan_dropped.to_string() + "\n")


y_val = val['T2M_difference'] 
X_val = val[['LC_CORINE','LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF','NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Define temporal and spatial features
temporal_feat = ['NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP', 'T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']
spatial_feat = ['LC_CORINE','LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF']

# Load the model
model = joblib.load("/kyukon/data/gent/vo/000/gvo00041/vsc46127/FEATURESEL/model_ALL.joblib")

# Create a figure
plt.figure(num=None, figsize=(20, 16), dpi=80, facecolor='w', edgecolor='k')

# Calculate feature importances
feat_importances = pd.Series(model.feature_importances_, index=X_val.columns)

# Sort feature importances and select top 27 features
top_feat_importances = feat_importances.nlargest(31).sort_values(ascending=True)

# Define colors for temporal and spatial features
colors = ['green' if feat in temporal_feat else 'orange' for feat in top_feat_importances.index]

# Plot feature importances
bars = top_feat_importances.plot(kind='barh', color=colors)

# Set the title and axis labels
plt.title('Feature Importance', fontsize=22)
plt.xlabel('Gini importance', fontsize=19)
#plt.ylabel('Features', fontsize=18)

# Create legend
temporal_patch = plt.Rectangle((0,0),1,1,fc="green", edgecolor = 'none')
spatial_patch = plt.Rectangle((0,0),1,1,fc='orange', edgecolor = 'none')
plt.legend([temporal_patch, spatial_patch], ['Temporal', 'Spatial'], fontsize=17, loc='lower right')

# Increase font size for better readability
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)

# Save the figure
plt.savefig('/kyukon/data/gent/vo/000/gvo00041/vsc46127/GINI_importance_ALL.png')





# Define temporal and spatial features
temporal_feat = ['NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP', 'T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']
spatial_feat = ['LC_CORINE','LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF']
# Create legend



def feature_color(feature):
    if feature in temporal_feat:
        return 'green'
    elif feature in spatial_feat:
        return 'orange'
    else:
        return 'gray'  # Use gray for features not in temporal or spatial lists

def plot_permutation_importance(clf, X, y):
    fig, ax = plt.subplots(figsize=(9, 12))  # Increase figure size
    result = permutation_importance(clf, X, y, n_repeats=5, random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()

    # Assign colors to each feature based on its type (temporal or spatial)
    colors = [feature_color(X.columns[i]) for i in perm_sorted_idx]

    ax.barh(
        y=X.columns[perm_sorted_idx],
        width=result.importances_mean[perm_sorted_idx],
        xerr=result.importances_std[perm_sorted_idx],  # Add error bars
        color=colors
    )
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_title("Permutation importance \n(Validation set)", fontsize=20)  # Increase title font size
    ax.set_xlabel("Decrease in mean square error", fontsize=16)  # Increase x-axis label font size
    ax.tick_params(axis='both', which='major', labelsize=15)  # Increase tick label font size
    fig.tight_layout()  # Adjust layout

    # Create legend
    temporal_patch = plt.Rectangle((0, 0), 1, 1, fc="green", edgecolor='none')
    spatial_patch = plt.Rectangle((0, 0), 1, 1, fc='orange', edgecolor='none')
    other_patch = plt.Rectangle((0, 0), 1, 1, fc='gray', edgecolor='none')
    ax.legend([temporal_patch, spatial_patch],
              ['Temporal', 'Spatial'], fontsize=17, loc='lower right')

    return fig, ax

fig, ax = plot_permutation_importance(model, X_val, y_val)
fig.savefig('/kyukon/data/gent/vo/000/gvo00041/vsc46127/permutation_importance_plot_ALL.png')


















