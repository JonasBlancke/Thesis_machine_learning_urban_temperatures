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
import seaborn as sns
import os


file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/all_data_FINAL.csv'
train = pd.read_csv(file_path, usecols=['T_2M_COR', 'T_TARGET', 'ruralurbanmask', 'city', 'hour', 'CBH', 'time'])
train = train.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})
train['T2M']=train['T2M']-273.15
train['T_TARGET']=train['T_TARGET']-273.15
train['T2M_difference']=train['T_TARGET'] - train['T2M']

train['CBH'] = train['CBH'].fillna(0)
# Drop all rows with NaN values in any column
train = train.dropna()


file_path='/kyukon/data/gent/vo/000/gvo00041/vsc46127/cluster/cities_all_FINAL.csv'
clusterval=pd.read_csv(file_path)
print(clusterval)

cluster1_cities= clusterval[clusterval['Cluster'] == 0]['City']
cluster2_cities= clusterval[clusterval['Cluster'] == 1]['City']
cluster3_cities= clusterval[clusterval['Cluster'] == 2]['City']
print(cluster1_cities)

train1 = train[train['city'].isin(cluster1_cities)]
train2 = train[train['city'].isin(cluster2_cities)]
train3 = train[train['city'].isin(cluster3_cities)]



import matplotlib.pyplot as plt
import pandas as pd
celcius = '\u00B0C'
# Assuming train1, train2, and train3 are DataFrames

for idx, train in enumerate([train1, train2, train3]):
    temp_train = train[['T2M', 'T_TARGET', 'ruralurbanmask', 'city', 'hour', 'time']]
    # Group the DataFrame by 'city' and 'hour' and calculate the temperature differences
    UHII_city_time = temp_train.groupby(['city', 'hour', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T_TARGET'].mean())
    mean_UHII_city_hour = UHII_city_time.groupby(['city', 'hour']).mean()
    mean_UHII = mean_UHII_city_hour.groupby('hour').mean()
    std_UHII = mean_UHII_city_hour.groupby('hour').std()


    ruralERA5_city_time = temp_train.groupby(['city', 'hour', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 1]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T2M'].mean())
    mean_ruralERA5_city_hour = ruralERA5_city_time.groupby(['city', 'hour']).mean()
    mean_ruralERA5 = mean_ruralERA5_city_hour.groupby('hour').mean()
    std_ruralERA5 = mean_ruralERA5_city_hour.groupby('hour').std()

    urbanERA5_city_time = temp_train.groupby(['city', 'hour', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 0]['T2M'].mean())
    mean_urbanERA5_city_hour = urbanERA5_city_time.groupby(['city', 'hour']).mean()
    mean_urbanERA5 = mean_urbanERA5_city_hour.groupby('hour').mean()
    std_urbanERA5 = mean_urbanERA5_city_hour.groupby('hour').std()




    #UHII = temp_train.groupby(['city', 'hour']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T_TARGET'].mean())
    #ruralERA5 = temp_train.groupby(['city', 'hour']).apply(lambda x: x[x['ruralurbanmask'] == 1]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T2M'].mean())
    #urbanERA5 = temp_train.groupby(['city', 'hour']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 0]['T2M'].mean())
    
    # Calculate mean and standard deviation for each hour
    #mean_UHII = UHII.groupby('hour').mean()
    #std_UHII = UHII.groupby('hour').std()
    
    #mean_ruralERA5 = ruralERA5.groupby('hour').mean()
    #std_ruralERA5 = ruralERA5.groupby('hour').std()
    
    #mean_urbanERA5 = urbanERA5.groupby('hour').mean()
    #std_urbanERA5 = urbanERA5.groupby('hour').std()
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot UHII with shifted error bars and line connecting points
    plt.errorbar(mean_UHII.index - 0.1, mean_UHII, yerr=std_UHII, label='UHII', fmt='o-', capsize=5)  # Shifted to the left
    
    # Plot ruralERA5 with line connecting points
    plt.errorbar(mean_ruralERA5.index, mean_ruralERA5, yerr=std_ruralERA5, label='T2M_difference (rural)', fmt='o-', capsize=5)  # Centered
    
    # Plot urbanERA5 with shifted error bars and line connecting points
    plt.errorbar(mean_urbanERA5.index + 0.1, mean_urbanERA5, yerr=std_urbanERA5, label='T2M_difference (urban)', fmt='o-', capsize=5)  # Shifted to the right
    
    # Add labels and title with increased font sizes
    plt.xlabel('Hour', fontsize=14)
    plt.ylabel(f'Temperature [{celcius}]', fontsize=14)
    plt.legend(loc='upper center', fontsize=12)
    plt.title(f'Cluster {idx+1}', fontsize=16)
    
    # Increase x-ticks font size
    plt.xticks(fontsize=12)
    
    # Save figure
    plt.grid(True)
    plt.savefig(f'figure_{idx+1}C.png')  # Save each figure with a unique name
    plt.close()

print("Figures saved successfully.")
