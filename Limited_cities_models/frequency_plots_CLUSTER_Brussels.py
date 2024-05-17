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
    'COAST': 'km',
    'ELEV': 'm',
    'POP': 'inhab/km\u00B2',  # Inhabitants per square kilometer
    'AHF': 'W/m\u00B2',  # Watts per square meter
    'ERA5_residual': '\u00B0C',
    'T_TARGET': '\u00B0C'  # Celsius


}



file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_data_FINAL.csv'
train_GENERAL_ALL= pd.read_csv(file_path, usecols=['BLH', 'T_2M_COR', 'T_TARGET', 'COAST', 'LC_CORINE', 'SP', 'SSR', 'PRECIP'])
train_GENERAL_ALL = train_GENERAL_ALL.rename(columns={'T_2M_COR': 'T2M'})
train_GENERAL_ALL['T2M']=train_GENERAL_ALL['T2M']-273.15
train_GENERAL_ALL['T_TARGET']=train_GENERAL_ALL['T_TARGET']-273.15
train_GENERAL_ALL['ERA5_residual']=train_GENERAL_ALL['T_TARGET'] - train_GENERAL_ALL['T2M']
train_GENERAL_ALL = train_GENERAL_ALL.dropna(subset=['T_TARGET'])

file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_data_BRUSSELS.csv'
train_BRUSSELS= pd.read_csv(file_path, usecols=['BLH', 'T_2M_COR', 'T_TARGET', 'COAST', 'LC_CORINE', 'SP', 'SSR', 'PRECIP'])
train_BRUSSELS = train_BRUSSELS.rename(columns={'T_2M_COR': 'T2M'})
train_BRUSSELS['T2M']=train_BRUSSELS['T2M']-273.15
train_BRUSSELS['T_TARGET']=train_BRUSSELS['T_TARGET']-273.15
train_BRUSSELS['ERA5_residual']=train_BRUSSELS['T_TARGET'] - train_BRUSSELS['T2M']
train_BRUSSELS = train_BRUSSELS.dropna(subset=['T_TARGET'])




file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER1_TEST_cities.csv'
train_CLUSTER1_TEST= pd.read_csv(file_path, usecols=['BLH', 'T_2M_COR', 'T_TARGET', 'COAST', 'LC_CORINE', 'SP', 'SSR', 'PRECIP'])
train_CLUSTER1_TEST = train_CLUSTER1_TEST.rename(columns={'T_2M_COR': 'T2M'})
train_CLUSTER1_TEST['T2M']=train_CLUSTER1_TEST['T2M']-273.15
train_CLUSTER1_TEST['T_TARGET']=train_CLUSTER1_TEST['T_TARGET']-273.15
train_CLUSTER1_TEST['ERA5_residual']=train_CLUSTER1_TEST['T_TARGET'] - train_CLUSTER1_TEST['T2M']
train_CLUSTER1_TEST = train_CLUSTER1_TEST.dropna(subset=['T_TARGET'])

file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER2_TEST_cities.csv'
train_CLUSTER2_TEST= pd.read_csv(file_path, usecols=['BLH', 'T_2M_COR', 'T_TARGET', 'COAST', 'LC_CORINE', 'SP', 'SSR', 'PRECIP'])
train_CLUSTER2_TEST = train_CLUSTER2_TEST.rename(columns={'T_2M_COR': 'T2M'})
train_CLUSTER2_TEST['T2M']=train_CLUSTER2_TEST['T2M']-273.15
train_CLUSTER2_TEST['T_TARGET']=train_CLUSTER2_TEST['T_TARGET']-273.15
train_CLUSTER2_TEST['ERA5_residual']=train_CLUSTER2_TEST['T_TARGET'] - train_CLUSTER2_TEST['T2M']
train_CLUSTER2_TEST = train_CLUSTER2_TEST.dropna(subset=['T_TARGET'])

file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/CLUSTER3_TEST_cities.csv'
train_CLUSTER3_TEST= pd.read_csv(file_path, usecols=['BLH', 'T_2M_COR', 'T_TARGET', 'COAST', 'LC_CORINE', 'SP', 'SSR', 'PRECIP'])
train_CLUSTER3_TEST = train_CLUSTER3_TEST.rename(columns={'T_2M_COR': 'T2M'})
train_CLUSTER3_TEST['T2M']=train_CLUSTER3_TEST['T2M']-273.15
train_CLUSTER3_TEST['T_TARGET']=train_CLUSTER3_TEST['T_TARGET']-273.15
train_CLUSTER3_TEST['ERA5_residual']=train_CLUSTER3_TEST['T_TARGET'] - train_CLUSTER3_TEST['T2M']
train_CLUSTER3_TEST = train_CLUSTER3_TEST.dropna(subset=['T_TARGET'])



label_mapping = {
    1: 'Urban',
    2: 'Sub-urban',
    3: 'Industrial',
    4: 'Urban green',
    5: 'Snow/Ice',
    6: 'Bare soil',
    7: 'Grassland',
    8: 'Cropland',
    9: 'Shrubland',
    10: 'Woodland',
    11: 'Broadleaf trees',
    12: 'Needleleaf trees',
    13: 'Rivers',
    14: 'Inland water bodies'
}



import seaborn as sns
import os

# Path to the folder for saving density plots
save_folder_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/'

# Check if the folder exists, if not, create it
os.makedirs(save_folder_path, exist_ok=True)




import matplotlib.pyplot as plt
import seaborn as sns
import os








# Define function to create grouped bar plots for each category
def create_grouped_bar_plot(train_data_list, labels, colors, save_path):
    plt.figure(figsize=(15, 6))
    bar_width = 0.15
    
    # Iterate through each dataset and plot its bars
    for i, (train_data, label, color) in enumerate(zip(train_data_list, labels, colors)):
        # Initialize category_counts with all 15 categories and set counts to 0
        category_counts = pd.Series(0, index=range(1, 15))
        x = np.arange(1, 15)  

        # Update counts for the categories that exist in the dataset
        category_counts.update(train_data['LC_CORINE'].value_counts(normalize=True) * 100)

        # Plot the bars for the current dataset
        plt.bar(x + i * bar_width, category_counts, bar_width, color=color, label=label)

    plt.xlabel('LC_CORINE', fontsize=16)
    plt.ylabel('Frequency [%]',fontsize=16)
    #plt.title('Histogram')

    plt.xticks(x + bar_width * len(train_data_list) / 2, labels=[label_mapping[x] for x in range(1, 15)], rotation=90, ha='center')
    plt.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, -0.5), shadow=True, ncol=3) # ncol = 3 to arrange the legend items in 3 columns
    plt.yticks(fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()
# Define colors for each dataset
colors = ['purple', 'yellow', 'blue', 'green', 'red']

# Define labels for each dataset
labels = ['GENERAL-60', 'Brussels', 'Test: Cluster 1', 'Test: Cluster 2', 'Test: Cluster 3']

# Define datasets to plot
train_data_list = [train_GENERAL_ALL, train_BRUSSELS, train_CLUSTER1_TEST, train_CLUSTER2_TEST, train_CLUSTER3_TEST]

# Specify save path
save_path = os.path.join(save_folder_path, f'LC_CORINE_percentage_grouped_bar_plot.png')

# Create grouped bar plot
create_grouped_bar_plot(train_data_list=train_data_list, labels=labels, colors=colors, save_path=save_path)




for column in ['COAST', 'LC_CORINE', 'SP']:
    plt.figure()
    sns.kdeplot(data=train_GENERAL_ALL[column], color='purple', label='GENERAL-60', shade=True, alpha=.3)
    sns.kdeplot(data=train_BRUSSELS[column], color='yellow', label='Brussels', shade=True, alpha=.3) 
    sns.kdeplot(data=train_CLUSTER1_TEST[column], color='blue', label='Test: Cluster 1', shade=True, alpha=.3) 
    sns.kdeplot(data=train_CLUSTER2_TEST[column], color='green', label='Test: Cluster 2', shade=True, alpha=.3) 
    sns.kdeplot(data=train_CLUSTER3_TEST[column], color='red', label='Test: Cluster 3', shade=True, alpha=.3) 

    # Add legend
    plt.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=3) # ncol = 3 to arrange the legend items in 3 columns

    # Add labels and title
    plt.xlabel(f'{column} [{column_units[column]}]',fontsize=16)
    plt.ylabel('Density',fontsize=16)
    #plt.title(f'Cluster 1',fontsize=16)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    # Show the plot (for debugging)
    plt.show()

    # Save the histogram plot
    save_path = os.path.join(save_folder_path, f'CL1_GENERAL_{column}_kde.png')
    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path)

    # Close the plot to prevent overlapping when creating the next one
    plt.close()

