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



file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_data_FINAL.csv'

train = pd.read_csv(file_path, usecols=['T_2M_COR', 'T_SK','CBH' ,'T_TARGET', 'city', 'hour','ruralurbanmask', 'COAST', 'POP', 'time'])


train = train.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})
train['T2M_difference']=train['T_TARGET'] - train['T2M']
train['T_TARGET']=train['T_TARGET']-273.15
train['T2M']=train['T2M']-273.15
train['SKT']=train['SKT']-273.15


train['CBH'] = train['CBH'].fillna(0)
# Drop all rows with NaN values in any column
train = train.dropna()



#X_train = train[['LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF','NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
#train_weather = train[['NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
#train_weather = train[['LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF', 'NDVI']]




#PER CITY:

temp_train=train[['T2M', 'T_TARGET', 'ruralurbanmask', 'city', 'time']]
# Group the DataFrame by 'city' and calculate the temperature difference
temp_difference_per_city = temp_train.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T_TARGET'].mean())
temp_difference_per_city=temp_difference_per_city.groupby(['city']).mean()
temp_difference_per_city.to_csv('UHII_difference_per_city_train.csv')

temp_difference_per_city = temp_train.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T2M'].mean() - x[x['ruralurbanmask'] == 1]['T2M'].mean())
temp_difference_per_city=temp_difference_per_city.groupby(['city']).mean()
temp_difference_per_city.to_csv('urbanminrural_difference_per_city_train.csv')

temp_difference_per_city = temp_train.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 1]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T2M'].mean())
temp_difference_per_city=temp_difference_per_city.groupby(['city']).mean()
temp_difference_per_city.to_csv('rural_difference_per_city_train.csv')

temp_difference_per_city = temp_train.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 0]['T2M'].mean())
temp_difference_per_city=temp_difference_per_city.groupby(['city']).mean()
temp_difference_per_city.to_csv('urban_difference_per_city_train.csv')

temp_difference_per_city = temp_train.groupby(['city', 'time']).apply(lambda x: x['T_TARGET'].mean() - x['T2M'].mean())
temp_difference_per_city=temp_difference_per_city.groupby(['city']).mean()
temp_difference_per_city.to_csv('general_difference_per_city_train.csv')



#PER CLUSTER:



import pandas as pd
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



# PER CLUSTER:

cluster1_cities = clusterval[clusterval['Cluster'] == 0]['City']
cluster2_cities = clusterval[clusterval['Cluster'] == 1]['City']
cluster3_cities = clusterval[clusterval['Cluster'] == 2]['City']

train_1 = train[train['city'].isin(cluster1_cities)]
train_2 = train[train['city'].isin(cluster2_cities)]
train_3 = train[train['city'].isin(cluster3_cities)]

# Calculate the temperature differences per cluster
temp_difference_per_cluster_1 = train1.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T_TARGET'].mean())
temp_difference_per_cluster_1=temp_difference_per_cluster_1.groupby(['city']).mean().mean()
temp_difference_per_cluster_2 = train2.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T_TARGET'].mean())
temp_difference_per_cluster_2=temp_difference_per_cluster_2.groupby(['city']).mean().mean()
temp_difference_per_cluster_3 = train3.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T_TARGET'].mean())
temp_difference_per_cluster_3=temp_difference_per_cluster_3.groupby(['city']).mean().mean()



#temp_difference_per_cluster_1 = train_1[train_1['ruralurbanmask'] == 0]['T_TARGET'].mean() - train_1[train_1['ruralurbanmask'] == 1]['T_TARGET'].mean()
#temp_difference_per_cluster_2 = train_2[train_2['ruralurbanmask'] == 0]['T_TARGET'].mean() - train_2[train_2['ruralurbanmask'] == 1]['T_TARGET'].mean()
#temp_difference_per_cluster_3 = train_3[train_3['ruralurbanmask'] == 0]['T_TARGET'].mean() - train_3[train_3['ruralurbanmask'] == 1]['T_TARGET'].mean()


# Create a DataFrame to store the results
temp_difference_clusters = pd.DataFrame({
    'Cluster': [1, 2, 3],
    'Temperature Difference': [temp_difference_per_cluster_1, temp_difference_per_cluster_2, temp_difference_per_cluster_3]
})

# Save the DataFrame to a CSV file
temp_difference_clusters.to_csv('UHII_difference_per_cluster_train.csv', index=False)




# Calculate the temperature differences per cluster

temp_difference_per_cluster_1 = train1.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T2M'].mean() - x[x['ruralurbanmask'] == 1]['T2M'].mean())
temp_difference_per_cluster_1=temp_difference_per_cluster_1.groupby(['city']).mean().mean()
temp_difference_per_cluster_2 = train2.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T2M'].mean() - x[x['ruralurbanmask'] == 1]['T2M'].mean())
temp_difference_per_cluster_2=temp_difference_per_cluster_2.groupby(['city']).mean().mean()
temp_difference_per_cluster_3 = train3.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T2M'].mean() - x[x['ruralurbanmask'] == 1]['T2M'].mean())
temp_difference_per_cluster_3=temp_difference_per_cluster_3.groupby(['city']).mean().mean()



#temp_difference_per_cluster_1 = train_1[train_1['ruralurbanmask'] == 0]['T2M'].mean() - train_1[train_1['ruralurbanmask'] == 1]['T2M'].mean()
#temp_difference_per_cluster_2 = train_2[train_2['ruralurbanmask'] == 0]['T2M'].mean() - train_2[train_2['ruralurbanmask'] == 1]['T2M'].mean()
#temp_difference_per_cluster_3 = train_3[train_3['ruralurbanmask'] == 0]['T2M'].mean() - train_3[train_3['ruralurbanmask'] == 1]['T2M'].mean()


# Create a DataFrame to store the results
temp_difference_clusters = pd.DataFrame({
    'Cluster': [1, 2, 3],
    'Temperature Difference': [temp_difference_per_cluster_1, temp_difference_per_cluster_2, temp_difference_per_cluster_3]
})

# Save the DataFrame to a CSV file
temp_difference_clusters.to_csv('urbanminrural_difference_per_cluster_train.csv', index=False)



# Calculate the temperature differences per cluster
temp_difference_per_cluster_1 = train1.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 1]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T2M'].mean())
temp_difference_per_cluster_1=temp_difference_per_cluster_1.groupby(['city']).mean().mean()
temp_difference_per_cluster_2 = train2.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 1]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T2M'].mean())
temp_difference_per_cluster_2=temp_difference_per_cluster_2.groupby(['city']).mean().mean()
temp_difference_per_cluster_3 = train3.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 1]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T2M'].mean())
temp_difference_per_cluster_3=temp_difference_per_cluster_3.groupby(['city']).mean().mean()


#temp_difference_per_cluster_1 = train_1[train_1['ruralurbanmask'] == 1]['T_TARGET'].mean() - train_1[train_1['ruralurbanmask'] == 1]['T2M'].mean()
#temp_difference_per_cluster_2 = train_2[train_2['ruralurbanmask'] == 1]['T_TARGET'].mean() - train_2[train_2['ruralurbanmask'] == 1]['T2M'].mean()
#temp_difference_per_cluster_3 = train_3[train_3['ruralurbanmask'] == 1]['T_TARGET'].mean() - train_3[train_3['ruralurbanmask'] == 1]['T2M'].mean()

# Create a DataFrame to store the results
temp_difference_clusters = pd.DataFrame({
    'Cluster': [1, 2, 3],
    'Temperature Difference': [temp_difference_per_cluster_1, temp_difference_per_cluster_2, temp_difference_per_cluster_3]
})

# Save the DataFrame to a CSV file
temp_difference_clusters.to_csv('rural_difference_per_cluster_train.csv', index=False)










# Calculate the temperature differences per cluster
temp_difference_per_cluster_1 = train1.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 0]['T2M'].mean())
temp_difference_per_cluster_1=temp_difference_per_cluster_1.groupby(['city']).mean().mean()
temp_difference_per_cluster_2 = train2.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 0]['T2M'].mean())
temp_difference_per_cluster_2=temp_difference_per_cluster_2.groupby(['city']).mean().mean()
temp_difference_per_cluster_3 = train3.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 0]['T2M'].mean())
temp_difference_per_cluster_3=temp_difference_per_cluster_3.groupby(['city']).mean().mean()



#temp_difference_per_cluster_1 = train_1[train_1['ruralurbanmask'] == 0]['T_TARGET'].mean() - train_1[train_1['ruralurbanmask'] == 0]['T2M'].mean()
#temp_difference_per_cluster_2 = train_2[train_2['ruralurbanmask'] == 0]['T_TARGET'].mean() - train_2[train_2['ruralurbanmask'] == 0]['T2M'].mean()
#temp_difference_per_cluster_3 = train_3[train_3['ruralurbanmask'] == 0]['T_TARGET'].mean() - train_3[train_3['ruralurbanmask'] == 0]['T2M'].mean()

# Create a DataFrame to store the results
temp_difference_clusters = pd.DataFrame({
    'Cluster': [1, 2, 3],
    'Temperature Difference': [temp_difference_per_cluster_1, temp_difference_per_cluster_2, temp_difference_per_cluster_3]
})

# Save the DataFrame to a CSV file
temp_difference_clusters.to_csv('urban_difference_per_cluster_train.csv', index=False)











# Calculate the temperature differences per cluster

temp_difference_per_cluster_1 = train1.groupby(['city', 'time']).apply(lambda x: x['T_TARGET'].mean() - x['T2M'].mean())
temp_difference_per_cluster_1=temp_difference_per_cluster_1.groupby(['city']).mean().mean()
temp_difference_per_cluster_2 = train2.groupby(['city', 'time']).apply(lambda x: x['T_TARGET'].mean() - x['T2M'].mean())
temp_difference_per_cluster_2=temp_difference_per_cluster_2.groupby(['city']).mean().mean()
temp_difference_per_cluster_3 = train3.groupby(['city', 'time']).apply(lambda x: x['T_TARGET'].mean() - x['T2M'].mean())
temp_difference_per_cluster_3=temp_difference_per_cluster_3.groupby(['city']).mean().mean()



#temp_difference_per_cluster_1 = train_1['T_TARGET'].mean() - train_1['T2M'].mean()
#temp_difference_per_cluster_2 = train_2['T_TARGET'].mean() - train_2['T2M'].mean()
#temp_difference_per_cluster_3 = train_3['T_TARGET'].mean() - train_3['T2M'].mean()

# Create a DataFrame to store the results
temp_difference_clusters = pd.DataFrame({
    'Cluster': [1, 2, 3],
    'Temperature Difference': [temp_difference_per_cluster_1, temp_difference_per_cluster_2, temp_difference_per_cluster_3]
})

# Save the DataFrame to a CSV file
temp_difference_clusters.to_csv('general_difference_per_cluster_train.csv', index=False)




#FOR ALL CITIES:

temp_train=train[['T2M', 'T_TARGET', 'ruralurbanmask', 'city', 'time']]

temp_difference = train.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T_TARGET'].mean())
temp_difference=temp_difference.groupby(['city']).mean().mean()
#temp_difference = temp_train[temp_train['ruralurbanmask'] == 0]['T_TARGET'].mean() - temp_train[temp_train['ruralurbanmask'] == 1]['T_TARGET'].mean()
temp_difference_df = pd.DataFrame({'temp_diff': [temp_difference]})
temp_difference_df.to_csv('UHII_difference_ALL_train.csv')

temp_difference = train.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T2M'].mean() - x[x['ruralurbanmask'] == 1]['T2M'].mean())
temp_difference=temp_difference.groupby(['city']).mean().mean()
#temp_difference = temp_train[temp_train['ruralurbanmask'] == 0]['T2M'].mean() - temp_train[temp_train['ruralurbanmask'] == 1]['T2M'].mean()
temp_difference_df = pd.DataFrame({'temp_diff': [temp_difference]})
temp_difference_df.to_csv('urbanminrural_difference_ALL_train.csv')

temp_difference = train.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 1]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T2M'].mean())
temp_difference=temp_difference.groupby(['city']).mean().mean()
#temp_difference = temp_train[temp_train['ruralurbanmask'] == 1]['T_TARGET'].mean() - temp_train[temp_train['ruralurbanmask'] == 1]['T2M'].mean()
temp_difference_df = pd.DataFrame({'temp_diff': [temp_difference]})
temp_difference_df.to_csv('rural_difference_ALL_train.csv')

temp_difference = train.groupby(['city', 'time']).apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 0]['T2M'].mean())
temp_difference=temp_difference.groupby(['city']).mean().mean()
#temp_difference = temp_train[temp_train['ruralurbanmask'] == 0]['T_TARGET'].mean() - temp_train[temp_train['ruralurbanmask'] == 0]['T2M'].mean()
temp_difference_df = pd.DataFrame({'temp_diff': [temp_difference]})
temp_difference_df.to_csv('urban_difference_ALL_train.csv')

temp_difference = train.groupby(['city', 'time']).apply(lambda x: x['T_TARGET'].mean() - x['T2M'].mean())
temp_difference=temp_difference.groupby(['city']).mean().mean()
#temp_difference = temp_train['T_TARGET'].mean() - temp_train['T2M'].mean()
temp_difference_df = pd.DataFrame({'temp_diff': [temp_difference]})
temp_difference_df.to_csv('general_difference_ALL_train.csv')










# Group data by city and calculate mean values
#grouped_data = train.groupby('city').mean()

# Create scatter plot for COAST vs T2M_difference
#plt.figure(figsize=(10, 6))
#sns.scatterplot(data=grouped_data, x='COAST', y='T2M_difference', hue=grouped_data.index, palette='Set1')

# Add city labels
#for city, x, y in zip(grouped_data.index, grouped_data['COAST'], grouped_data['T2M_difference']):
#    plt.text(x, y, city, fontsize=8, ha='right')

#plt.title('COAST vs T2M_difference')
#plt.xlabel('COAST')
#plt.ylabel('T2M_difference')
#plt.legend(title='City')
#plt.grid(True)
#plt.savefig('COAST_vs_T2M_difference.png')
#plt.show()

# Create scatter plot for POP vs T2M_difference
#plt.figure(figsize=(10, 6))
#sns.scatterplot(data=grouped_data, x='POP', y='T2M_difference', hue=grouped_data.index, palette='Set1')

# Add city labels
#for city, x, y in zip(grouped_data.index, grouped_data['POP'], grouped_data['T2M_difference']):
#    plt.text(x, y, city, fontsize=8, ha='right')

#plt.title('POP vs T2M_difference')
#plt.xlabel('POP')
#plt.ylabel('T2M_difference')
#plt.grid(True)
#plt.savefig('POP_vs_T2M_difference.png')
#plt.show()



# Calculate mean T2M_difference and UHII per city
#mean_T2M_difference = train.groupby('city')['T2M_difference'].mean()
#mean_UHII = train.groupby('city').apply(lambda x: x[x['ruralurbanmask'] == 0]['T_TARGET'].mean() - x[x['ruralurbanmask'] == 1]['T_TARGET'].mean())

# Create DataFrame with mean values
#mean_data = pd.DataFrame({'mean_T2M_difference': mean_T2M_difference, 'mean_UHII': mean_UHII})



# Create scatter plot for mean_UHII vs mean_T2M_difference
#plt.figure(figsize=(10, 6))
#sns.scatterplot(data=mean_data, x='mean_UHII', y='mean_T2M_difference', hue=mean_data.index, palette='Set1')

# Add city labels
#for city, x, y in zip(mean_data.index, mean_data['mean_UHII'], mean_data['mean_T2M_difference']):
#    plt.text(x, y, city, fontsize=8, ha='right')

#plt.title('Mean UHII vs Mean T2M_difference')
#plt.xlabel('Mean UHII')
#plt.ylabel('Mean T2M_difference')
#plt.grid(True)
#plt.savefig('Mean_UHII_vs_Mean_T2M_difference.png')
#plt.show()

