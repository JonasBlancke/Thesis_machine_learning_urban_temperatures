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




cities=['Brussels']
years = ['2014']  # Add your desired years here
months = ['01', '02', '03', '04','05','06','07','08','09','10','11', '12']
city_data=[]
for year in years:
    for month in months:
        for city in cities:
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
                               'CAPE', 'RH', 'WS', 
                                'POP', 'HEIGHT', 'COAST', 
                                'LC_CORINE', 'IMPERV', 'ELEV', 
                               'GEOPOT',  'T_2M_SL', 'DECL', 'SOLAR_ELEV', 
                                'T_2M_COR']

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
val_data_merged.to_csv('validation_data_BRUSSELS.csv', index=False)




years = ['2008', '2009', '2010', '2011', '2012','2013']  # Add your desired years here
months = ['01', '02', '03', '04','05','06','07','08','09','10','11', '12']
city_data=[]
for year in years:
    for month in months:
        for city in cities:
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
                               'CAPE', 'RH', 'WS', 
                                'POP', 'HEIGHT', 'COAST', 
                                'LC_CORINE', 'IMPERV', 'ELEV', 
                               'GEOPOT',  'T_2M_SL', 'DECL', 'SOLAR_ELEV', 
                                'T_2M_COR']

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
train_data_merged.to_csv('train_data_BRUSSELS.csv', index=False)



years = ['2015', '2016', '2017']  # Add your desired years here
months = ['01', '02', '03', '04','05','06','07','08','09','10','11', '12']
city_data=[]
for year in years:
    for month in months:
        for city in cities:
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
                               'CAPE', 'RH', 'WS', 
                                'POP', 'HEIGHT', 'COAST', 
                                'LC_CORINE', 'IMPERV', 'ELEV', 
                               'GEOPOT',  'T_2M_SL', 'DECL', 'SOLAR_ELEV', 
                                'T_2M_COR']

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
test_data_merged.to_csv('test_data_BRUSSELS.csv', index=False)



















