#import xarray as xr
#import rioxarray as rxr
import os
import numpy as np  
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error
import joblib
import time

import pandas as pd

file_path='/kyukon/data/gent/vo/000/gvo00041/vsc46127/cluster/cities_train_FINAL.csv'
train_cities=pd.read_csv(file_path)
train_cities_name=train_cities['City']
print(train_cities_name)


#train_cities0=train_cities[train_cities['Cluster'] == 0]
#cities_train_0=train_cities0['City']
#train_cities1=train_cities[train_cities['Cluster'] == 1]
#cities_train_1=train_cities1['City']
#train_cities2=train_cities[train_cities['Cluster'] == 2]
#cities_train_2=train_cities2['City']


file_path='/kyukon/data/gent/vo/000/gvo00041/vsc46127/cluster/cities_validation_FINAL.csv'
val_cities=pd.read_csv(file_path)
validation_cities_name=val_cities['City']
print(validation_cities_name)

#val_cities0=val_cities[val_cities['Cluster'] == 0]
#cities_val_0=val_cities0['City']
#val_cities1=val_cities[val_cities['Cluster'] == 1]
#cities_val_1=val_cities1['City']
#val_cities2=val_cities[val_cities['Cluster'] == 2]
#cities_val_2=val_cities2['City']



file_path='/kyukon/data/gent/vo/000/gvo00041/vsc46127/cluster/cities_test_FINAL.csv'
test_cities=pd.read_csv(file_path)
test_cities_name=test_cities['City']


# Concatenate all city names into a single DataFrame
all_cities = pd.concat([train_cities_name, validation_cities_name, test_cities_name], ignore_index=True)
# Optionally, if you want to remove duplicates
all_cities_name = all_cities.drop_duplicates().reset_index(drop=True)
# Display the DataFrame
print(all_cities_name)

years = ['2014']  # Add your desired years here
months = ['01', '02', '03', '04','05','06','07','08','09','10','11', '12']
city_data=[]
for year in years:
    for month in months:
        for city in validation_cities_name:
            start_time = time.perf_counter()
            file_path = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_test/MODEL_PART1/ready_for_model_{city}_{year}_{month}.csv'
            data = pd.read_csv(file_path, low_memory=False)
                       # Define columns to convert to integers
            int_columns = ['hour', 'day', 'month', 'year', 'ruralurbanmask', 'landseamask', 'LCZ', 'local_time']

            # Check for NaN values in specified columns
            nan_counts = data[int_columns].isna().sum()

            # Print count of NaN values for each column if NaN values are found
            if nan_counts.any():
                for col, nan_count in nan_counts.items():
                    print(f"NaN values in column '{col}': {nan_count}")

                # Skip rows with NaN values in the specified columns
                data = data.dropna(subset=int_columns)

            # Convert columns to integers outside of the loop
            data[int_columns] = data[int_columns].astype('int32')


            float32_columns = ['index', 'SP', 'y', 'x', 'BLH', 'PRECIP', 'T_TARGET', 'T_2M', 'SSR', 'TCC', 
                               'CAPE', 'RH', 'U10', 'wind_speed', 'V10', 'T_SK', 'SM', 'CBH', 'WS', 
                               'BUILT_FRAC', 'POP', 'NDVI', 'HEIGHT', 'COAST', 'LC_TREE', 'LC_WATER', 
                               'LC_BUILT', 'LC_BAREANDVEG', 'LC_CORINE', 'AHF', 'IMPERV', 'ELEV', 
                               'GEOPOT',  'T_2M_SL', 'DECL', 'SOLAR_ELEV', 
                                'WD', 'T_2M_COR']

            data[float32_columns] = data[float32_columns].astype('float32')

            # Ensure 'city' column remains as string
            data['city'] = data['city'].astype(str)

            city_data.append(data)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            with open('run_time.txt', 'a') as f:
                f.write(f"Time for adding {city} {year} {month}: {elapsed_time} seconds\n")

# Concatenate all data for the current city
val_data_merged = pd.concat(city_data, axis=0, ignore_index=True)
val_data_merged.to_csv('validation_data_FINAL.csv', index=False)




years = ['2008', '2009', '2010', '2011', '2012','2013']  # Add your desired years here
months = ['01', '02', '03', '04','05','06','07','08','09','10','11', '12']
city_data=[]
for year in years:
    for month in months:
        for city in train_cities_name:
            start_time = time.perf_counter()
            file_path = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_test/MODEL_PART1/ready_for_model_{city}_{year}_{month}.csv'
            data = pd.read_csv(file_path, low_memory=False)
                       # Define columns to convert to integers
            int_columns = ['hour', 'day', 'month', 'year', 'ruralurbanmask', 'landseamask', 'LCZ', 'local_time']

            # Check for NaN values in specified columns
            nan_counts = data[int_columns].isna().sum()

            # Print count of NaN values for each column if NaN values are found
            if nan_counts.any():
                for col, nan_count in nan_counts.items():
                    print(f"NaN values in column '{col}': {nan_count}")

                # Skip rows with NaN values in the specified columns
                data = data.dropna(subset=int_columns)

            # Convert columns to integers outside of the loop
            data[int_columns] = data[int_columns].astype('int32')


            float32_columns = ['index', 'SP', 'y', 'x', 'BLH', 'PRECIP', 'T_TARGET', 'T_2M', 'SSR', 'TCC', 
                               'CAPE', 'RH', 'U10', 'wind_speed', 'V10', 'T_SK', 'SM', 'CBH', 'WS', 
                               'BUILT_FRAC', 'POP', 'NDVI', 'HEIGHT', 'COAST', 'LC_TREE', 'LC_WATER', 
                               'LC_BUILT', 'LC_BAREANDVEG', 'LC_CORINE', 'AHF', 'IMPERV', 'ELEV', 
                               'GEOPOT',  'T_2M_SL', 'DECL', 'SOLAR_ELEV', 
                                'WD', 'T_2M_COR']

            data[float32_columns] = data[float32_columns].astype('float32')

            # Ensure 'city' column remains as string
            data['city'] = data['city'].astype(str)
            city_data.append(data)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            with open('run_time.txt', 'a') as f:
                f.write(f"Time for adding {city} {year} {month}: {elapsed_time} seconds\n")

# Concatenate all data for the current city
train_data_merged = pd.concat(city_data, axis=0, ignore_index=True)
train_data_merged.to_csv('train_data_FINAL.csv', index=False)



years = ['2015', '2016', '2017']  # Add your desired years here
months = ['01', '02', '03', '04','05','06','07','08','09','10','11', '12']
city_data=[]
for year in years:
    for month in months:
        for city in test_cities_name:
            start_time = time.perf_counter()
            file_path = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_test/MODEL_PART1/ready_for_model_{city}_{year}_{month}.csv'
            data = pd.read_csv(file_path, low_memory=False)
                        # Define columns to convert to integers
            int_columns = ['hour', 'day', 'month', 'year', 'ruralurbanmask', 'landseamask', 'LCZ', 'local_time']

            # Check for NaN values in specified columns
            nan_counts = data[int_columns].isna().sum()

            # Print count of NaN values for each column if NaN values are found
            if nan_counts.any():
                for col, nan_count in nan_counts.items():
                    print(f"NaN values in column '{col}': {nan_count}")

                # Skip rows with NaN values in the specified columns
                data = data.dropna(subset=int_columns)

            # Convert columns to integers outside of the loop
            data[int_columns] = data[int_columns].astype('int32')


            float32_columns = ['index', 'SP', 'y', 'x', 'BLH', 'PRECIP', 'T_TARGET', 'T_2M', 'SSR', 'TCC', 
                               'CAPE', 'RH', 'U10', 'wind_speed', 'V10', 'T_SK', 'SM', 'CBH', 'WS', 
                               'BUILT_FRAC', 'POP', 'NDVI', 'HEIGHT', 'COAST', 'LC_TREE', 'LC_WATER', 
                               'LC_BUILT', 'LC_BAREANDVEG', 'LC_CORINE', 'AHF', 'IMPERV', 'ELEV', 
                               'GEOPOT',  'T_2M_SL', 'DECL', 'SOLAR_ELEV', 
                                'WD', 'T_2M_COR']

            data[float32_columns] = data[float32_columns].astype('float32')

            # Ensure 'city' column remains as string
            data['city'] = data['city'].astype(str)
            city_data.append(data)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            with open('run_time.txt', 'a') as f:
                f.write(f"Time for adding {city} {year} {month}: {elapsed_time} seconds\n")

# Concatenate all data for the current city
test_data_merged = pd.concat(city_data, axis=0, ignore_index=True)
test_data_merged.to_csv('test_data_FINAL.csv', index=False)








