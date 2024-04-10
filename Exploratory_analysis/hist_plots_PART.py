# -*- coding: utf-8 -*-
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

import codecs
# Define the column_units dictionary with Unicode escape sequences
# Define the column_units dictionary with Unicode escape sequences
column_units = {
    'NDVI': '-',
    'SKT': '\u00B0C',  # Celsius
    'SM': 'm\u00B3/m\u00B3',  # Cubic meter per cubic meter
    'RH': '%',
    'SP': 'Pa',
    'PRECIP': 'mm',
    'T2M': '\u00B0C',  # Celsius
    'WS': 'm/s',
    'WD': '\u00B0',    # Degree symbol for wind direction
    'U10': 'm/s',
    'V10': 'm/s',
    'TCC': '-',
    'CBH': 'm',
    'CAPE': 'J/kg',
    'BLH': 'm',
    'SSR': 'W/m\u00B2',  # Watts per square meter
    'SOLAR_ELEV': '\u00B0',  # Degree symbol for solar elevation
    'DECL': '\u00B0',         # Degree symbol for declination
    'T2M_difference': '\u00B0C',  # Celsius
    'LC_CORINE': '-',
    'LC_WATER': '-',
    'LC_TREE': '-',
    'LC_BAREANDVEG': '-',
    'LC_BUILT': '-',
    'IMPERV': '%',
    'HEIGHT': 'm',
    'BUILT_FRAC': '-',
    'COAST': 'm',
    'ELEV': 'm',
    'POP': 'inhab/km\u00B2',  # Inhabitants per square kilometer
    'AHF': 'W/m\u00B2'  # Watts per square meter
}








file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_data_FINAL.csv'

train= pd.read_csv(file_path)
train = train.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})

#X_train = train[['LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF','NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
#train_weather = train[['NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
#train_spatial = train[['LC_CORINE', 'LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF', 'NDVI']]


import seaborn as sns
import os

# Path to the folder for saving density plots
save_folder_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/'

# Check if the folder exists, if not, create it
os.makedirs(save_folder_path, exist_ok=True)


file_path='/kyukon/data/gent/vo/000/gvo00041/vsc46127/cluster/cities_train_FINAL.csv'
clusterval=pd.read_csv(file_path)
print(clusterval)

cluster1_cities= clusterval[clusterval['Cluster'] == 0]['City']
cluster2_cities= clusterval[clusterval['Cluster'] == 1]['City']
cluster3_cities= clusterval[clusterval['Cluster'] == 2]['City']
print(cluster1_cities)



train['T2M_difference']=train['T_TARGET'] - train['T2M']
train['T2M']=train['T2M']-273.15
train['SKT']=train['SKT']-273.15

train_1 = train[train['city'].isin(cluster1_cities)]
print(train_1)
train_2 = train[train['city'].isin(cluster2_cities)]
train_3 = train[train['city'].isin(cluster3_cities)]

#X_train = train[['LC_CORINE','LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF','NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL', 'T2M_difference']]
train_1_spatial = train_1[['COAST', 'ELEV', 'POP', 'AHF', 'NDVI']]


train_2_spatial = train_2[[ 'COAST', 'ELEV', 'POP', 'AHF', 'NDVI']]


train_3_spatial = train_3[[ 'COAST', 'ELEV', 'POP', 'AHF', 'NDVI']]




print("Total NaN values per column in train_1_spatial:")
print(train_1_spatial.isna().sum())

print("Total NaN values per column in train_2_spatial:")
print(train_2_spatial.isna().sum())

print("Total NaN values per column in train_3_spatial:")
print(train_3_spatial.isna().sum())


# Loop through each variable and create histograms for train_1_spatial
for column in train_1_spatial.columns:
    plt.figure()
    #sns.histplot(train_1_spatial[column], bins=50, color='blue', label='train 1', alpha=0.5)
    #sns.histplot(train_2_spatial[column], bins=50, color='green', label='train 2', alpha=0.5)
    #sns.histplot(train_3_spatial[column], bins=50, color='red', label='train 3', alpha=0.5)
    sns.kdeplot(data=train_1_spatial[column], color='blue', label='Cluster 1', shade=True)
    sns.kdeplot(data=train_2_spatial[column], color='green', label='Cluster 2', shade=True)
    sns.kdeplot(data=train_3_spatial[column], color='red', label='Cluster 3', shade=True)


    # Add legend
    plt.legend()

    # Add labels and title
    plt.xlabel(f'{column} [{column_units[column]}]')
    plt.ylabel('Density')
    #plt.title(f'Histogram of {column} for train 1, train 2, and train 3')

    # Show the plot (for debugging)
    plt.show()

    # Save the histogram plot
    save_path = os.path.join(save_folder_path, f'train_1_spatial_{column}_kde.png')
    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path)
    
    # Close the plot to prevent overlapping when creating the next one
    plt.close()





