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
val = pd.read_csv(file_path, usecols=['T_2M', 'city','T_TARGET','T_2M_COR', 'LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP','NDVI', 'T_SK', 'SM', 'RH', 'SP', 'PRECIP', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL'])


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
X_val = val[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP','NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Define temporal and spatial features
temporal_feat = ['NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP', 'T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']
spatial_feat = ['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP']

# Load the model
model = joblib.load("/kyukon/data/gent/vo/000/gvo00041/vsc46127/FEATURESEL/model_REDUNDANT.joblib")

# Create a figure
plt.figure(num=None, figsize=(20, 16), dpi=80, facecolor='w', edgecolor='k')

# Calculate feature importances
feat_importances = pd.Series(model.feature_importances_, index=X_val.columns)

# Sort feature importances and select top 27 features
top_feat_importances = feat_importances.nlargest(27)

# Define colors for temporal and spatial features
colors = ['green' if feat in temporal_feat else 'orange' for feat in top_feat_importances.index]

# Plot feature importances
bars = top_feat_importances.plot(kind='barh', color=colors)

# Set the title and axis labels
plt.title('Feature Importance', fontsize=25)
plt.xlabel('Importance', fontsize=22)
plt.ylabel('Features', fontsize=22)

# Create legend
temporal_patch = plt.Rectangle((0,0),1,1,fc="green", edgecolor = 'none')
spatial_patch = plt.Rectangle((0,0),1,1,fc='orange', edgecolor = 'none')
plt.legend([temporal_patch, spatial_patch], ['Temporal', 'Spatial'], fontsize=16)

# Increase font size for better readability
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Save the figure
plt.savefig('/kyukon/data/gent/vo/000/gvo00041/vsc46127/feature_importance_REDUNDANT.png')




# ERA5 corrected
y_pred = val['T2M']
y_val=val['T2M_difference'] +val['T2M']



# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
pearson_corr, _ = pearsonr(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
bias = np.mean(y_pred - y_val)

# Create a DataFrame for the general validation indices
general_indices = pd.DataFrame({
    'RMSE': [rmse],
    'Pearson Correlation': [pearson_corr],
    'MAE': [mae],
    'R2 Score': [r2],
    'Bias': [bias]
})

# Save the DataFrame to a CSV file
general_indices.to_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/evaluation_results_ERA5_CORRECTED_general_REDUNDANT.csv", index=False)



# ERA5 not corrected


y_pred = val['T2M_NC']
y_val=val['T2M_difference'] +val['T2M']



# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
pearson_corr, _ = pearsonr(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
bias = np.mean(y_pred - y_val)

# Create a DataFrame for the general validation indices
general_indices = pd.DataFrame({
    'RMSE': [rmse],
    'Pearson Correlation': [pearson_corr],
    'MAE': [mae],
    'R2 Score': [r2],
    'Bias': [bias]
})

# Save the DataFrame to a CSV file
general_indices.to_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/evaluation_results_ERA5_NONCORRECTED_general_REDUNDANT.csv", index=False)









# Make predictions
y_pred = model.predict(X_val)+X_val['T2M']
y_val=val['T2M_difference'] +X_val['T2M']



# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
pearson_corr, _ = pearsonr(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)
bias = np.mean(y_pred - y_val)

# Create a DataFrame for the general validation indices
general_indices = pd.DataFrame({
    'RMSE': [rmse],
    'Pearson Correlation': [pearson_corr],
    'MAE': [mae],
    'R2 Score': [r2],
    'Bias': [bias]
})

# Save the DataFrame to a CSV file
general_indices.to_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/evaluation_results_REDUNDANT_general.csv", index=False)





# Calculate evaluation metrics for each city
evaluation_results = {}
cities = val['city'].unique()
y_val = val['T2M_difference'] 

# Iterate over each city
for city in cities:
    # Get indices corresponding to the current city
    city_val = val[val['city'] == city]
    X_val_city=city_val[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP','NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
    y_val_city=city_val['T2M_difference']
    # Predictions for the current city
    city_y_pred = model.predict(X_val_city) + X_val_city['T2M']
    city_y_val = city_val['T2M_difference'] +X_val_city['T2M']

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(city_y_val, city_y_pred))
    pearson_corr, _ = pearsonr(city_y_val, city_y_pred)
    mae = mean_absolute_error(city_y_val, city_y_pred)
    r2 = r2_score(city_y_val, city_y_pred)
    bias = np.mean(city_y_pred - city_y_val)
    
    # Store evaluation results for the current city
    evaluation_results[city] = {'RMSE': rmse, 'Pearson Correlation': pearson_corr, 'MAE': mae, 'R2 Score': r2, 'Bias': bias}

# Convert evaluation results to a DataFrame
evaluation_df = pd.DataFrame.from_dict(evaluation_results, orient='index')

# Save evaluation results to a CSV file
evaluation_df.to_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/evaluation_results_REDUNDANT_percity.csv")

# Print confirmation message
print("Evaluation results saved successfully.")



def plot_permutation_importance(clf, X, y):
    fig, ax = plt.subplots(figsize=(7, 6))  # Create a new figure and axis
    result = permutation_importance(clf, X, y, n_repeats=5, random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()

    ax.boxplot(
        result.importances[perm_sorted_idx].T,
        vert=False,
        labels=X.columns[perm_sorted_idx],
    )
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_title("Permutation Importances \n(validation set)")
    ax.set_xlabel("Decrease in accuracy score")
    fig.tight_layout()  # Adjust layout
    return fig, ax

fig, ax = plot_permutation_importance(model, X_val, y_val)

fig.savefig('/kyukon/data/gent/vo/000/gvo00041/vsc46127/REDUNDANT_permutation_importance_plot.png')







