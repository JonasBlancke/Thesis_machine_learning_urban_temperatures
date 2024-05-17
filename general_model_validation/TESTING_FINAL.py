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

file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/test_data_FINAL.csv'
test = pd.read_csv(file_path, usecols=['T_2M', 'city','T_TARGET','T_2M_COR', 'LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL'])


# Count NaN values per column before dropping
test_nan_before_drop = test.isnull().sum()

# Drop rows with NaN values in 'T_TARGET' column
test = test.dropna(subset=['T_TARGET'])

test = test.rename(columns={'T_2M': 'T2M_NC'})
test = test.rename(columns={'T_2M_COR': 'T2M'})

test['T2M_difference'] = test['T_TARGET'] - test['T2M']
test['T2M'] = test['T2M'] - 273.15
test['T2M_NC'] = test['T2M_NC'] - 273.15

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
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Define temporal and spatial features
temporal_feat = [ 'RH', 'SP', 'PRECIP', 'T2M', 'WS','TCC',  'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']
spatial_feat = ['LC_CORINE', 'IMPERV', 'HEIGHT',  'COAST', 'ELEV', 'POP']

# Load the model
model = joblib.load("/kyukon/data/gent/vo/000/gvo00041/vsc46127/FEATURESEL/model_FINAL.joblib")




def plot_permutation_importance(clf, X, y):
    fig, ax = plt.subplots(figsize=(9, 12))  # Increase figure size
    result = permutation_importance(clf, X, y, n_repeats=5, random_state=42)
    perm_sorted_idx = result.importances_mean.argsort()

    ax.barh(
        y=X.columns[perm_sorted_idx],
        width=result.importances_mean[perm_sorted_idx],
        xerr=result.importances_std[perm_sorted_idx],  # Add error bars
        color='skyblue'
    )
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_title("Permutation Importances \n(Test set)", fontsize=20)  # Increase title font size
    ax.set_xlabel("Decrease in accuracy score", fontsize=16)  # Increase x-axis label font size
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increase tick label font size
    fig.tight_layout()  # Adjust layout
    return fig, ax

# Assuming you have already defined your model (model), X_test, and y_test
fig, ax = plot_permutation_importance(model, X_test, y_test)

# Save the figure
fig.savefig('/kyukon/data/gent/vo/000/gvo00041/vsc46127/FINAL_permutation_importance_plot_TEST.png')











# Create a figure
plt.figure(num=None, figsize=(20, 16), dpi=80, facecolor='w', edgecolor='k')

# Calculate feature importances
feat_importances = pd.Series(model.feature_importances_, index=X_test.columns)

# Sort feature importances and select top 27 features
top_feat_importances = feat_importances.nlargest(31)

# Define colors for temporal and spatial features
colors = ['green' if feat in temporal_feat else 'orange' for feat in top_feat_importances.index]

# Plot feature importances
bars = top_feat_importances.plot(kind='barh', color=colors)

# Set the title and axis labels
plt.title('Feature Importance', fontsize=20)
plt.xlabel('Importance', fontsize=18)
plt.ylabel('Features', fontsize=18)

# Create legend
temporal_patch = plt.Rectangle((0,0),1,1,fc="green", edgecolor = 'none')
spatial_patch = plt.Rectangle((0,0),1,1,fc='orange', edgecolor = 'none')
plt.legend([temporal_patch, spatial_patch], ['Temporal', 'Spatial'], fontsize=16)

# Increase font size for better readability
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Save the figure
plt.savefig('/kyukon/data/gent/vo/000/gvo00041/vsc46127/feature_importance_FINAL_TEST.png')




# ERA5 corrected
y_pred = test['T2M']
y_test=test['T2M_difference'] +test['T2M']



# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
pearson_corr, _ = pearsonr(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
bias = np.mean(y_pred - y_test)

# Create a DataFrame for the general validation indices
general_indices = pd.DataFrame({
    'RMSE': [rmse],
    'Pearson Correlation': [pearson_corr],
    'MAE': [mae],
    'R2 Score': [r2],
    'Bias': [bias]
})

# Save the DataFrame to a CSV file
general_indices.to_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/evaluation_results_ERA5_CORRECTED_general_FINAL_TEST.csv", index=False)



# ERA5 not corrected


y_pred = test['T2M_NC']
y_test=test['T2M_difference'] +test['T2M']



# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
pearson_corr, _ = pearsonr(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
bias = np.mean(y_pred - y_test)

# Create a DataFrame for the general validation indices
general_indices = pd.DataFrame({
    'RMSE': [rmse],
    'Pearson Correlation': [pearson_corr],
    'MAE': [mae],
    'R2 Score': [r2],
    'Bias': [bias]
})

# Save the DataFrame to a CSV file
general_indices.to_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/evaluation_results_ERA5_NONCORRECTED_general_FINAL_TEST.csv", index=False)









# Make predictions
y_pred = model.predict(X_test)+X_test['T2M']
y_test=test['T2M_difference'] +X_test['T2M']



# Evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
pearson_corr, _ = pearsonr(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
bias = np.mean(y_pred - y_test)

# Create a DataFrame for the general validation indices
general_indices = pd.DataFrame({
    'RMSE': [rmse],
    'Pearson Correlation': [pearson_corr],
    'MAE': [mae],
    'R2 Score': [r2],
    'Bias': [bias]
})

# Save the DataFrame to a CSV file
general_indices.to_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/evaluation_results_FINAL_general_TEST.csv", index=False)





# Calculate evaluation metrics for each city
evaluation_results = {}
cities = test['city'].unique()
y_test = test['T2M_difference'] 

# Iterate over each city
for city in cities:
    # Get indices corresponding to the current city
    city_test = test[test['city'] == city]
    X_test_city=city_test[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
    y_test_city=city_test['T2M_difference']
    # Predictions for the current city
    city_y_pred = model.predict(X_test_city) + X_test_city['T2M']
    city_y_test = city_test['T2M_difference'] +X_test_city['T2M']

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
    pearson_corr, _ = pearsonr(city_y_test, city_y_pred)
    mae = mean_absolute_error(city_y_test, city_y_pred)
    r2 = r2_score(city_y_test, city_y_pred)
    bias = np.mean(city_y_pred - city_y_test)
    
    # Store evaluation results for the current city
    evaluation_results[city] = {'RMSE': rmse, 'Pearson Correlation': pearson_corr, 'MAE': mae, 'R2 Score': r2, 'Bias': bias}

# Convert evaluation results to a DataFrame
evaluation_df = pd.DataFrame.from_dict(evaluation_results, orient='index')

# Save evaluation results to a CSV file
evaluation_df.to_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/evaluation_results_FINAL_percity_TEST.csv")

# Print confirmation message
print("Evaluation results saved successfully.")





# Calculate evaluation metrics for each city
evaluation_results = {}
cities = test['city'].unique()
y_test = test['T2M_difference'] 

# Iterate over each city
for city in cities:
    # Get indices corresponding to the current city
    city_test = test[test['city'] == city]
    #X_test_city=city_test[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
    y_test_city=city_test['T2M_difference']
    # Predictions for the current city
    city_y_pred = city_test['T2M']
    city_y_test = city_test['T2M_difference'] +city_test['T2M']

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
    pearson_corr, _ = pearsonr(city_y_test, city_y_pred)
    mae = mean_absolute_error(city_y_test, city_y_pred)
    r2 = r2_score(city_y_test, city_y_pred)
    bias = np.mean(city_y_pred - city_y_test)
    
    # Store evaluation results for the current city
    evaluation_results[city] = {'RMSE': rmse, 'Pearson Correlation': pearson_corr, 'MAE': mae, 'R2 Score': r2, 'Bias': bias}

# Convert evaluation results to a DataFrame
evaluation_df = pd.DataFrame.from_dict(evaluation_results, orient='index')

# Save evaluation results to a CSV file
evaluation_df.to_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/ERA5CORRECTED_results_FINAL_percity_TEST.csv")

# Print confirmation message
print("Evaluation results saved successfully.")


























