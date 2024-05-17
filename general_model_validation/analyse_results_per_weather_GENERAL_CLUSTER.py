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



file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/test_data_FINAL.csv'
test = pd.read_csv(file_path, usecols=['T_SK','T_2M', 'city','T_TARGET','T_2M_COR', 'LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP', 'WS', 'WD',  'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL'])
cluster_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/cluster/cities_test_FINAL.csv'
cluster_data = pd.read_csv(cluster_path)
cluster_data['Cluster']=cluster_data['Cluster']+1
cluster_data = cluster_data.rename(columns={'City': 'city'})
test = pd.merge(cluster_data, test, on='city')

# Drop rows with NaN values in 'T_TARGET' column
test = test.dropna(subset=['T_TARGET'])

test = test.rename(columns={'T_2M': 'T2M_NC'})
test = test.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})

test['T2M_difference'] = test['T_TARGET'] - test['T2M']
test['T2M'] = test['T2M'] - 273.15
test['SKT'] = test['SKT'] - 273.15
test['T2M_NC'] = test['T2M_NC'] - 273.15

test = test.dropna(subset=['T2M_difference'])

for column in test.columns:
    if test[column].isnull().any():
        test[column] = test.groupby('city')[column].transform(lambda x: x.fillna(x.median()))


start_time = time.perf_counter()



model = joblib.load("/kyukon/data/gent/vo/000/gvo00041/vsc46127/FEATURESEL/model_FINAL.joblib")



test=test[test['Cluster']==1]

situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['PRECIP'] >=0.001)]
if len(situation_data) > 0:

   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test +X_test['T2M']

   # Calculate evaluation metrics
   situation1_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation1_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test +X_test['T2M']

   situation1_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation1_era_bias = np.mean(city_y_pred - city_y_test)
    
else:
   situation1_model_bias = np.nan
   situation1_model_rmse = np.nan
   situation1_era_bias = np.nan
   situation1_era_rmse = np.nan



# Situation 2
situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['PRECIP'] < 0.001)]
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation2_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation2_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation2_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation2_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation2_model_bias = np.nan
   situation2_model_rmse = np.nan
   situation2_era_bias = np.nan
   situation2_era_rmse = np.nan



situation3_data = test[(test['SOLAR_ELEV'] < 0) & (test['PRECIP'] >=0.001)]
# Situation 3
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation3_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation3_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation3_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation3_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation3_model_bias = np.nan
   situation3_model_rmse = np.nan
   situation3_era_bias = np.nan
   situation3_era_rmse = np.nan



situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['PRECIP'] < 0.001)]
# Situation 4
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation4_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation4_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation4_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation4_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation4_model_bias = np.nan
   situation4_model_rmse = np.nan
   situation4_era_bias = np.nan
   situation4_era_rmse = np.nan


situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['TCC'] >= 0.5)]
# Situation 5
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation5_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation5_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation5_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation5_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation5_model_bias = np.nan
   situation5_model_rmse = np.nan
   situation5_era_bias = np.nan
   situation5_era_rmse = np.nan

# Repeat the same process for other situations (situation6, situation7, ..., situation12)


situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['TCC'] < 0.5)]
# Situation 6
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation6_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation6_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation6_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation6_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation6_model_bias = np.nan
   situation6_model_rmse = np.nan
   situation6_era_bias = np.nan
   situation6_era_rmse = np.nan


situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['TCC'] >= 0.5)]
# Situation 7
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation7_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation7_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation7_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation7_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation7_model_bias = np.nan
   situation7_model_rmse = np.nan
   situation7_era_bias = np.nan
   situation7_era_rmse = np.nan

# Continue this pattern for the remaining situations (situation8, situation9, ..., situation12)


situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['TCC'] < 0.5)]
# Situation 8
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation8_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation8_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation8_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation8_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation8_model_bias = np.nan
   situation8_model_rmse = np.nan
   situation8_era_bias = np.nan
   situation8_era_rmse = np.nan


situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['WS'] >=4)]
# Situation 9
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation9_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation9_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation9_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation9_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation9_model_bias = np.nan
   situation9_model_rmse = np.nan
   situation9_era_bias = np.nan
   situation9_era_rmse = np.nan

# Continue this pattern for the remaining situations (situation10, situation11, and situation12)

situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['WS'] < 4)]
# Situation 10
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation10_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation10_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation10_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation10_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation10_model_bias = np.nan
   situation10_model_rmse = np.nan
   situation10_era_bias = np.nan
   situation10_era_rmse = np.nan

situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['WS'] >=4)]
# Situation 11
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation11_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation11_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation11_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation11_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation11_model_bias = np.nan
   situation11_model_rmse = np.nan
   situation11_era_bias = np.nan
   situation11_era_rmse = np.nan

# Continue this pattern for the remaining situation (situation12)

situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['WS'] < 4)]
# Situation 12
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation12_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation12_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation12_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation12_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation12_model_bias = np.nan
   situation12_model_rmse = np.nan
   situation12_era_bias = np.nan
   situation12_era_rmse = np.nan




end_time = time.perf_counter()
elapsed_time = end_time - start_time
with open('run_time2.txt', 'a') as f:
    f.write(f"Time: {elapsed_time} seconds\n")



# Create a dictionary to store evaluation results
evaluation_results = {
    'situation1_model_rmse': situation1_model_rmse,
    'situation1_model_bias': situation1_model_bias,
    'situation1_era_rmse': situation1_era_rmse,
    'situation1_era_bias': situation1_era_bias,
    'situation2_model_rmse': situation2_model_rmse,
    'situation2_model_bias': situation2_model_bias,
    'situation2_era_rmse': situation2_era_rmse,
    'situation2_era_bias': situation2_era_bias,
    'situation3_model_rmse': situation3_model_rmse,
    'situation3_model_bias': situation3_model_bias,
    'situation3_era_rmse': situation3_era_rmse,
    'situation3_era_bias': situation3_era_bias,
    'situation4_model_rmse': situation4_model_rmse,
    'situation4_model_bias': situation4_model_bias,
    'situation4_era_rmse': situation4_era_rmse,
    'situation4_era_bias': situation4_era_bias,
    'situation5_model_rmse': situation5_model_rmse,
    'situation5_model_bias': situation5_model_bias,
    'situation5_era_rmse': situation5_era_rmse,
    'situation5_era_bias': situation5_era_bias,
    'situation6_model_rmse': situation6_model_rmse,
    'situation6_model_bias': situation6_model_bias,
    'situation6_era_rmse': situation6_era_rmse,
    'situation6_era_bias': situation6_era_bias,
    'situation7_model_rmse': situation7_model_rmse,
    'situation7_model_bias': situation7_model_bias,
    'situation7_era_rmse': situation7_era_rmse,
    'situation7_era_bias': situation7_era_bias,
    'situation8_model_rmse': situation8_model_rmse,
    'situation8_model_bias': situation8_model_bias,
    'situation8_era_rmse': situation8_era_rmse,
    'situation8_era_bias': situation8_era_bias,
    'situation9_model_rmse': situation9_model_rmse,
    'situation9_model_bias': situation9_model_bias,
    'situation9_era_rmse': situation9_era_rmse,
    'situation9_era_bias': situation9_era_bias,
    'situation10_model_rmse': situation10_model_rmse,
    'situation10_model_bias': situation10_model_bias,
    'situation10_era_rmse': situation10_era_rmse,
    'situation10_era_bias': situation10_era_bias,
    'situation11_model_rmse': situation11_model_rmse,
    'situation11_model_bias': situation11_model_bias,
    'situation11_era_rmse': situation11_era_rmse,
    'situation11_era_bias': situation11_era_bias,
    'situation12_model_rmse': situation12_model_rmse,
    'situation12_model_bias': situation12_model_bias,
    'situation12_era_rmse': situation12_era_rmse,
    'situation12_era_bias': situation12_era_bias
}

# Convert the dictionary to a DataFrame
evaluation_df = pd.DataFrame.from_dict(evaluation_results, orient='index')

# Save the DataFrame to a CSV file
evaluation_df.to_csv('evaluation_results_CLUSTER1.csv')







#--------------------------------------------------------------------------------------------------------------------------------------------------




file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/test_data_FINAL.csv'
test = pd.read_csv(file_path, usecols=['T_SK','T_2M', 'city','T_TARGET','T_2M_COR', 'LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP', 'WS', 'WD',  'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL'])
cluster_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/cluster/cities_test_FINAL.csv'
cluster_data = pd.read_csv(cluster_path)
cluster_data['Cluster']=cluster_data['Cluster']+1
cluster_data = cluster_data.rename(columns={'City': 'city'})
test = pd.merge(cluster_data, test, on='city')

# Drop rows with NaN values in 'T_TARGET' column
test = test.dropna(subset=['T_TARGET'])

test = test.rename(columns={'T_2M': 'T2M_NC'})
test = test.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})

test['T2M_difference'] = test['T_TARGET'] - test['T2M']
test['T2M'] = test['T2M'] - 273.15
test['SKT'] = test['SKT'] - 273.15
test['T2M_NC'] = test['T2M_NC'] - 273.15

test = test.dropna(subset=['T2M_difference'])

for column in test.columns:
    if test[column].isnull().any():
        test[column] = test.groupby('city')[column].transform(lambda x: x.fillna(x.median()))


start_time = time.perf_counter()



model = joblib.load("/kyukon/data/gent/vo/000/gvo00041/vsc46127/FEATURESEL/model_FINAL.joblib")



test=test[test['Cluster']==2]

situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['PRECIP'] >=0.001)]
if len(situation_data) > 0:

   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test +X_test['T2M']

   # Calculate evaluation metrics
   situation1_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation1_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test +X_test['T2M']

   situation1_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation1_era_bias = np.mean(city_y_pred - city_y_test)
    
else:
   situation1_model_bias = np.nan
   situation1_model_rmse = np.nan
   situation1_era_bias = np.nan
   situation1_era_rmse = np.nan



# Situation 2
situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['PRECIP'] < 0.001)]
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation2_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation2_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation2_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation2_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation2_model_bias = np.nan
   situation2_model_rmse = np.nan
   situation2_era_bias = np.nan
   situation2_era_rmse = np.nan



situation3_data = test[(test['SOLAR_ELEV'] < 0) & (test['PRECIP'] >=0.001)]
# Situation 3
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation3_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation3_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation3_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation3_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation3_model_bias = np.nan
   situation3_model_rmse = np.nan
   situation3_era_bias = np.nan
   situation3_era_rmse = np.nan



situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['PRECIP'] < 0.001)]
# Situation 4
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation4_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation4_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation4_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation4_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation4_model_bias = np.nan
   situation4_model_rmse = np.nan
   situation4_era_bias = np.nan
   situation4_era_rmse = np.nan


situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['TCC'] >= 0.5)]
# Situation 5
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation5_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation5_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation5_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation5_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation5_model_bias = np.nan
   situation5_model_rmse = np.nan
   situation5_era_bias = np.nan
   situation5_era_rmse = np.nan

# Repeat the same process for other situations (situation6, situation7, ..., situation12)


situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['TCC'] < 0.5)]
# Situation 6
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation6_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation6_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation6_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation6_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation6_model_bias = np.nan
   situation6_model_rmse = np.nan
   situation6_era_bias = np.nan
   situation6_era_rmse = np.nan


situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['TCC'] >= 0.5)]
# Situation 7
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation7_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation7_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation7_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation7_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation7_model_bias = np.nan
   situation7_model_rmse = np.nan
   situation7_era_bias = np.nan
   situation7_era_rmse = np.nan

# Continue this pattern for the remaining situations (situation8, situation9, ..., situation12)


situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['TCC'] < 0.5)]
# Situation 8
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation8_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation8_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation8_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation8_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation8_model_bias = np.nan
   situation8_model_rmse = np.nan
   situation8_era_bias = np.nan
   situation8_era_rmse = np.nan


situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['WS'] >=4)]
# Situation 9
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation9_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation9_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation9_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation9_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation9_model_bias = np.nan
   situation9_model_rmse = np.nan
   situation9_era_bias = np.nan
   situation9_era_rmse = np.nan

# Continue this pattern for the remaining situations (situation10, situation11, and situation12)

situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['WS'] < 4)]
# Situation 10
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation10_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation10_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation10_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation10_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation10_model_bias = np.nan
   situation10_model_rmse = np.nan
   situation10_era_bias = np.nan
   situation10_era_rmse = np.nan

situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['WS'] >=4)]
# Situation 11
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation11_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation11_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation11_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation11_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation11_model_bias = np.nan
   situation11_model_rmse = np.nan
   situation11_era_bias = np.nan
   situation11_era_rmse = np.nan

# Continue this pattern for the remaining situation (situation12)

situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['WS'] < 4)]
# Situation 12
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation12_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation12_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation12_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation12_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation12_model_bias = np.nan
   situation12_model_rmse = np.nan
   situation12_era_bias = np.nan
   situation12_era_rmse = np.nan




end_time = time.perf_counter()
elapsed_time = end_time - start_time
with open('run_time2.txt', 'a') as f:
    f.write(f"Time: {elapsed_time} seconds\n")



# Create a dictionary to store evaluation results
evaluation_results = {
    'situation1_model_rmse': situation1_model_rmse,
    'situation1_model_bias': situation1_model_bias,
    'situation1_era_rmse': situation1_era_rmse,
    'situation1_era_bias': situation1_era_bias,
    'situation2_model_rmse': situation2_model_rmse,
    'situation2_model_bias': situation2_model_bias,
    'situation2_era_rmse': situation2_era_rmse,
    'situation2_era_bias': situation2_era_bias,
    'situation3_model_rmse': situation3_model_rmse,
    'situation3_model_bias': situation3_model_bias,
    'situation3_era_rmse': situation3_era_rmse,
    'situation3_era_bias': situation3_era_bias,
    'situation4_model_rmse': situation4_model_rmse,
    'situation4_model_bias': situation4_model_bias,
    'situation4_era_rmse': situation4_era_rmse,
    'situation4_era_bias': situation4_era_bias,
    'situation5_model_rmse': situation5_model_rmse,
    'situation5_model_bias': situation5_model_bias,
    'situation5_era_rmse': situation5_era_rmse,
    'situation5_era_bias': situation5_era_bias,
    'situation6_model_rmse': situation6_model_rmse,
    'situation6_model_bias': situation6_model_bias,
    'situation6_era_rmse': situation6_era_rmse,
    'situation6_era_bias': situation6_era_bias,
    'situation7_model_rmse': situation7_model_rmse,
    'situation7_model_bias': situation7_model_bias,
    'situation7_era_rmse': situation7_era_rmse,
    'situation7_era_bias': situation7_era_bias,
    'situation8_model_rmse': situation8_model_rmse,
    'situation8_model_bias': situation8_model_bias,
    'situation8_era_rmse': situation8_era_rmse,
    'situation8_era_bias': situation8_era_bias,
    'situation9_model_rmse': situation9_model_rmse,
    'situation9_model_bias': situation9_model_bias,
    'situation9_era_rmse': situation9_era_rmse,
    'situation9_era_bias': situation9_era_bias,
    'situation10_model_rmse': situation10_model_rmse,
    'situation10_model_bias': situation10_model_bias,
    'situation10_era_rmse': situation10_era_rmse,
    'situation10_era_bias': situation10_era_bias,
    'situation11_model_rmse': situation11_model_rmse,
    'situation11_model_bias': situation11_model_bias,
    'situation11_era_rmse': situation11_era_rmse,
    'situation11_era_bias': situation11_era_bias,
    'situation12_model_rmse': situation12_model_rmse,
    'situation12_model_bias': situation12_model_bias,
    'situation12_era_rmse': situation12_era_rmse,
    'situation12_era_bias': situation12_era_bias
}

# Convert the dictionary to a DataFrame
evaluation_df = pd.DataFrame.from_dict(evaluation_results, orient='index')

# Save the DataFrame to a CSV file
evaluation_df.to_csv('evaluation_results_CLUSTER2.csv')





#-------------------------------------------------------------------------------------------------------------------------------------------------------




file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/test_data_FINAL.csv'
test = pd.read_csv(file_path, usecols=['T_SK','T_2M', 'city','T_TARGET','T_2M_COR', 'LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP', 'WS', 'WD',  'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL'])
cluster_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/cluster/cities_test_FINAL.csv'
cluster_data = pd.read_csv(cluster_path)
cluster_data['Cluster']=cluster_data['Cluster']+1
cluster_data = cluster_data.rename(columns={'City': 'city'})
test = pd.merge(cluster_data, test, on='city')

# Drop rows with NaN values in 'T_TARGET' column
test = test.dropna(subset=['T_TARGET'])

test = test.rename(columns={'T_2M': 'T2M_NC'})
test = test.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})

test['T2M_difference'] = test['T_TARGET'] - test['T2M']
test['T2M'] = test['T2M'] - 273.15
test['SKT'] = test['SKT'] - 273.15
test['T2M_NC'] = test['T2M_NC'] - 273.15

test = test.dropna(subset=['T2M_difference'])

for column in test.columns:
    if test[column].isnull().any():
        test[column] = test.groupby('city')[column].transform(lambda x: x.fillna(x.median()))


start_time = time.perf_counter()



model = joblib.load("/kyukon/data/gent/vo/000/gvo00041/vsc46127/FEATURESEL/model_FINAL.joblib")



test=test[test['Cluster']==3]

situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['PRECIP'] >=0.001)]
if len(situation_data) > 0:

   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test +X_test['T2M']

   # Calculate evaluation metrics
   situation1_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation1_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test +X_test['T2M']

   situation1_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation1_era_bias = np.mean(city_y_pred - city_y_test)
    
else:
   situation1_model_bias = np.nan
   situation1_model_rmse = np.nan
   situation1_era_bias = np.nan
   situation1_era_rmse = np.nan



# Situation 2
situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['PRECIP'] < 0.001)]
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation2_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation2_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation2_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation2_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation2_model_bias = np.nan
   situation2_model_rmse = np.nan
   situation2_era_bias = np.nan
   situation2_era_rmse = np.nan



situation3_data = test[(test['SOLAR_ELEV'] < 0) & (test['PRECIP'] >=0.001)]
# Situation 3
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation3_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation3_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation3_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation3_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation3_model_bias = np.nan
   situation3_model_rmse = np.nan
   situation3_era_bias = np.nan
   situation3_era_rmse = np.nan



situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['PRECIP'] < 0.001)]
# Situation 4
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation4_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation4_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation4_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation4_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation4_model_bias = np.nan
   situation4_model_rmse = np.nan
   situation4_era_bias = np.nan
   situation4_era_rmse = np.nan


situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['TCC'] >= 0.5)]
# Situation 5
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation5_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation5_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation5_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation5_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation5_model_bias = np.nan
   situation5_model_rmse = np.nan
   situation5_era_bias = np.nan
   situation5_era_rmse = np.nan

# Repeat the same process for other situations (situation6, situation7, ..., situation12)


situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['TCC'] < 0.5)]
# Situation 6
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation6_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation6_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation6_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation6_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation6_model_bias = np.nan
   situation6_model_rmse = np.nan
   situation6_era_bias = np.nan
   situation6_era_rmse = np.nan


situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['TCC'] >= 0.5)]
# Situation 7
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation7_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation7_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation7_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation7_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation7_model_bias = np.nan
   situation7_model_rmse = np.nan
   situation7_era_bias = np.nan
   situation7_era_rmse = np.nan

# Continue this pattern for the remaining situations (situation8, situation9, ..., situation12)


situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['TCC'] < 0.5)]
# Situation 8
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation8_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation8_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation8_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation8_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation8_model_bias = np.nan
   situation8_model_rmse = np.nan
   situation8_era_bias = np.nan
   situation8_era_rmse = np.nan


situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['WS'] >=4)]
# Situation 9
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation9_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation9_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation9_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation9_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation9_model_bias = np.nan
   situation9_model_rmse = np.nan
   situation9_era_bias = np.nan
   situation9_era_rmse = np.nan

# Continue this pattern for the remaining situations (situation10, situation11, and situation12)

situation_data = test[(test['SOLAR_ELEV'] > 0) & (test['WS'] < 4)]
# Situation 10
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation10_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation10_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation10_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation10_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation10_model_bias = np.nan
   situation10_model_rmse = np.nan
   situation10_era_bias = np.nan
   situation10_era_rmse = np.nan

situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['WS'] >=4)]
# Situation 11
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation11_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation11_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation11_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation11_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation11_model_bias = np.nan
   situation11_model_rmse = np.nan
   situation11_era_bias = np.nan
   situation11_era_rmse = np.nan

# Continue this pattern for the remaining situation (situation12)

situation_data = test[(test['SOLAR_ELEV'] < 0) & (test['WS'] < 4)]
# Situation 12
if len(situation_data) > 0:
   y_test = situation_data['T2M_difference'] 
   X_test = situation_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
   # Predictions 
   city_y_pred = model.predict(X_test) + X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   # Calculate evaluation metrics
   situation12_model_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation12_model_bias = np.mean(city_y_pred - city_y_test)

   city_y_pred = X_test['T2M']
   city_y_test = y_test + X_test['T2M']

   situation12_era_rmse = np.sqrt(mean_squared_error(city_y_test, city_y_pred))
   situation12_era_bias = np.mean(city_y_pred - city_y_test)
else:
   situation12_model_bias = np.nan
   situation12_model_rmse = np.nan
   situation12_era_bias = np.nan
   situation12_era_rmse = np.nan




end_time = time.perf_counter()
elapsed_time = end_time - start_time
with open('run_time2.txt', 'a') as f:
    f.write(f"Time: {elapsed_time} seconds\n")



# Create a dictionary to store evaluation results
evaluation_results = {
    'situation1_model_rmse': situation1_model_rmse,
    'situation1_model_bias': situation1_model_bias,
    'situation1_era_rmse': situation1_era_rmse,
    'situation1_era_bias': situation1_era_bias,
    'situation2_model_rmse': situation2_model_rmse,
    'situation2_model_bias': situation2_model_bias,
    'situation2_era_rmse': situation2_era_rmse,
    'situation2_era_bias': situation2_era_bias,
    'situation3_model_rmse': situation3_model_rmse,
    'situation3_model_bias': situation3_model_bias,
    'situation3_era_rmse': situation3_era_rmse,
    'situation3_era_bias': situation3_era_bias,
    'situation4_model_rmse': situation4_model_rmse,
    'situation4_model_bias': situation4_model_bias,
    'situation4_era_rmse': situation4_era_rmse,
    'situation4_era_bias': situation4_era_bias,
    'situation5_model_rmse': situation5_model_rmse,
    'situation5_model_bias': situation5_model_bias,
    'situation5_era_rmse': situation5_era_rmse,
    'situation5_era_bias': situation5_era_bias,
    'situation6_model_rmse': situation6_model_rmse,
    'situation6_model_bias': situation6_model_bias,
    'situation6_era_rmse': situation6_era_rmse,
    'situation6_era_bias': situation6_era_bias,
    'situation7_model_rmse': situation7_model_rmse,
    'situation7_model_bias': situation7_model_bias,
    'situation7_era_rmse': situation7_era_rmse,
    'situation7_era_bias': situation7_era_bias,
    'situation8_model_rmse': situation8_model_rmse,
    'situation8_model_bias': situation8_model_bias,
    'situation8_era_rmse': situation8_era_rmse,
    'situation8_era_bias': situation8_era_bias,
    'situation9_model_rmse': situation9_model_rmse,
    'situation9_model_bias': situation9_model_bias,
    'situation9_era_rmse': situation9_era_rmse,
    'situation9_era_bias': situation9_era_bias,
    'situation10_model_rmse': situation10_model_rmse,
    'situation10_model_bias': situation10_model_bias,
    'situation10_era_rmse': situation10_era_rmse,
    'situation10_era_bias': situation10_era_bias,
    'situation11_model_rmse': situation11_model_rmse,
    'situation11_model_bias': situation11_model_bias,
    'situation11_era_rmse': situation11_era_rmse,
    'situation11_era_bias': situation11_era_bias,
    'situation12_model_rmse': situation12_model_rmse,
    'situation12_model_bias': situation12_model_bias,
    'situation12_era_rmse': situation12_era_rmse,
    'situation12_era_bias': situation12_era_bias
}

# Convert the dictionary to a DataFrame
evaluation_df = pd.DataFrame.from_dict(evaluation_results, orient='index')

# Save the DataFrame to a CSV file
evaluation_df.to_csv('evaluation_results_CLUSTER3.csv')











