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

file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER2_TEST_cities.csv'
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



experiment = '2cities_CL2'
# Load the model using the formatted file path
model = joblib.load(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/model_{experiment}.joblib")

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
general_indices.to_csv(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/CLUSTER2/CL2_GENERAL_{experiment}.csv", index=False)

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
evaluation_df.to_csv(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/CLUSTER2/CL2_percity_{experiment}.csv")
# Print confirmation message
print("Evaluation results saved successfully.")




#--------------------------------------------------------------------------------------------------------------------------------------------------------



experiment = '5cities_CL2'
# Load the model using the formatted file path
model = joblib.load(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/model_{experiment}.joblib")

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
general_indices.to_csv(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/CLUSTER2/CL2_GENERAL_{experiment}.csv", index=False)

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
evaluation_df.to_csv(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/CLUSTER2/CL2_percity_{experiment}.csv")
# Print confirmation message
print("Evaluation results saved successfully.")







#--------------------------------------------------------------------------------------------------------------------------------------------------------



experiment = 'GENERAL_6cities'
# Load the model using the formatted file path
model = joblib.load(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/{experiment}.joblib")

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
general_indices.to_csv(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/CLUSTER2/CL2_GENERAL_{experiment}.csv", index=False)

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
evaluation_df.to_csv(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/CLUSTER2/CL2_percity_{experiment}.csv")
# Print confirmation message
print("Evaluation results saved successfully.")



#-------------------------------------------------------------------------------------------------------------------------------------------



experiment = 'GENERAL_15cities'
# Load the model using the formatted file path
model = joblib.load(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/{experiment}.joblib")

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
general_indices.to_csv(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/CLUSTER2/CL2_GENERAL_{experiment}.csv", index=False)

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
evaluation_df.to_csv(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/CLUSTER2/CL2_percity_{experiment}.csv")
# Print confirmation message
print("Evaluation results saved successfully.")


#-------------------------------------------------------------------------------------------------------------------------------------------


experiment = 'GENERAL_ALLcities'
# Load the model using the formatted file path
model = joblib.load(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/{experiment}.joblib")

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
general_indices.to_csv(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/CLUSTER2/CL2_GENERAL_{experiment}.csv", index=False)

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
evaluation_df.to_csv(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/CLUSTER2/CL2_percity_{experiment}.csv")
# Print confirmation message
print("Evaluation results saved successfully.")









#--------------------------------------------------------------------------------------------------------------------------------------------------------


experiment = 'ALLcities_CL2'
# Load the model using the formatted file path
model = joblib.load(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/model_{experiment}.joblib")

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
general_indices.to_csv(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/CLUSTER2/CL2_GENERAL_{experiment}.csv", index=False)

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
evaluation_df.to_csv(f"/kyukon/data/gent/vo/000/gvo00041/vsc46127/PART2/CLUSTER2/CL2_percity_{experiment}.csv")
# Print confirmation message
print("Evaluation results saved successfully.")



