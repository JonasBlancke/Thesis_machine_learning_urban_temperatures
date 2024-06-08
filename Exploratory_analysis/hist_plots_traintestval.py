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
    'PRECIP': 'mm/h',
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
train= pd.read_csv(file_path, usecols=['city','T_TARGET', 'LC_CORINE', 'LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF', 'NDVI','NDVI', 'T_SK', 'SM', 'RH', 'SP', 'PRECIP','T_2M_COR', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL'])
train = train.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})
train['T2M_difference']=train['T_TARGET'] - train['T2M']
train['T2M']=train['T2M']-273.15
train['SKT']=train['SKT']-273.15
# Count NaN values per column before dropping
train_nan_before_drop = train.isnull().sum()
# Drop rows with NaN values in 'T_TARGET' column
train = train.dropna(subset=['T_TARGET'])
train = train.dropna(subset=['T2M_difference'])
# Count NaN values per column after dropping 'T_TARGET'
train_nan_after_drop = train.isnull().sum()
# Fill NaN values with 0 in 'CBH' column
# Count NaN values per column after filling with 0
train_nan_after_fill = train.isnull().sum()
start_time = time.perf_counter()
# Fill NaN values with the median value of the specific 'City'
for column in train.columns:
    if train[column].isnull().any():
        train[column] = train.groupby('city')[column].transform(lambda x: x.fillna(x.median()))



file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/validation_data_FINAL.csv'
validation= pd.read_csv(file_path, usecols=['city','T_TARGET', 'LC_CORINE', 'LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF', 'NDVI','NDVI', 'T_SK', 'SM', 'RH', 'SP', 'PRECIP','T_2M_COR', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL'])
validation = validation.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})
validation['T2M_difference']=validation['T_TARGET'] - validation['T2M']
validation['T2M']=validation['T2M']-273.15
validation['SKT']=validation['SKT']-273.15
# Count NaN values per column before dropping
validation_nan_before_drop = validation.isnull().sum()
# Drop rows with NaN values in 'T_TARGET' column
validation = validation.dropna(subset=['T_TARGET'])
validation = validation.dropna(subset=['T2M_difference'])
# Count NaN values per column after dropping 'T_TARGET'
validation_nan_after_drop = validation.isnull().sum()
# Fill NaN values with 0 in 'CBH' column
# Count NaN values per column after filling with 0
validation_nan_after_fill = validation.isnull().sum()
start_time = time.perf_counter()
# Fill NaN values with the median value of the specific 'City'
for column in validation.columns:
    if validation[column].isnull().any():
        validation[column] = validation.groupby('city')[column].transform(lambda x: x.fillna(x.median()))



file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/test_data_FINAL.csv'
test= pd.read_csv(file_path, usecols=['city','T_TARGET','LC_CORINE', 'LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF', 'NDVI','NDVI', 'T_SK', 'SM', 'RH', 'SP', 'PRECIP','T_2M_COR', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL'])
test = test.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})
test['T2M_difference']=test['T_TARGET'] - test['T2M']
test['T2M']=test['T2M']-273.15
test['SKT']=test['SKT']-273.15
# Count NaN values per column before dropping
test_nan_before_drop = test.isnull().sum()
# Drop rows with NaN values in 'T_TARGET' column
test = test.dropna(subset=['T_TARGET'])
test = test.rename(columns={'T_2M_COR': 'T2M'})
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




import seaborn as sns
import os

# Path to the folder for saving density plots
save_folder_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/'

# Check if the folder exists, if not, create it
os.makedirs(save_folder_path, exist_ok=True)



columns = ['T2M_difference', 'LC_CORINE', 'LC_WATER', 'LC_TREE', 'LC_BAREANDVEG', 'LC_BUILT', 'IMPERV', 'HEIGHT', 'BUILT_FRAC', 'COAST', 'ELEV', 'POP', 'AHF', 'NDVI', 'SKT', 'SM', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']
columns = ['SKT', 'SM', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'WD', 'U10', 'V10', 'TCC', 'CBH', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']
columns=['LC_CORINE']
# Define the font size for all text elements
label_font_size = 16
legend_font_size = 14
tick_font_size = 14



# Define the font size for all text elements
label_font_size = 16
legend_font_size = 14
tick_font_size = 14
small_tick_font_size = 10  # Smaller font size for SP x-ticks

for column in columns:
    plt.figure()

    # Clip values based on the specified conditions
    if column == 'IMPERV':
        train_column = train[column]
        test_column = test[column]
        validation_column = validation[column]

        sns.kdeplot(data=train_column, color='yellow', label='Train', shade=True)
        sns.kdeplot(data=test_column, color='green', label='Test', shade=True)
        sns.kdeplot(data=validation_column, color='purple', label='Validation', shade=True)
        plt.xlim(0, 100)
        # Add legend
        plt.legend(fontsize=legend_font_size)
        # Add labels and title
        plt.xlabel(f'{column} [{column_units[column]}]', fontsize=label_font_size)
        plt.ylabel('Density', fontsize=label_font_size)
        # Adjust tick parameters
        if column == 'SP':
            plt.xticks(fontsize=small_tick_font_size)
        else:
            plt.xticks(fontsize=tick_font_size)
        plt.yticks(fontsize=tick_font_size)


        # Ensure layout is adjusted to prevent text from overlapping
        plt.tight_layout()
    elif column in ['POP', 'PRECIP', 'CAPE']:
        # Apply log10 transformation
        train_column = np.log10(train[column] + 1)  # Add 1 to avoid log(0)
        test_column = np.log10(test[column] + 1)
        validation_column = np.log10(validation[column] + 1)

        sns.kdeplot(data=train_column, color='yellow', label='Train', shade=True)
        sns.kdeplot(data=test_column, color='green', label='Test', shade=True)
        sns.kdeplot(data=validation_column, color='purple', label='Validation', shade=True)

        # Add legend
        plt.legend(fontsize=legend_font_size)
        # Add labels and title
        plt.xlabel(f'log10({column}) [{column_units[column]}]', fontsize=label_font_size)
        plt.ylabel('Density', fontsize=label_font_size)
        # Adjust tick parameters


        # Ensure layout is adjusted to prevent text from overlapping
        plt.tight_layout()
    
    elif column == 'LC_CORINE':
        # Plot density by categorical variable
        train_counts = train[column].value_counts(normalize=True)*100
        test_counts = test[column].value_counts(normalize=True)*100
        validation_counts = validation[column].value_counts(normalize=True)*100

        categories = sorted(train_counts.index.union(test_counts.index).union(validation_counts.index))
        x = np.arange(len(categories))

        bar_width = 0.25

        plt.bar(x - bar_width, train_counts.reindex(categories, fill_value=0), color='yellow', width=bar_width, label='Train')
        plt.bar(x, test_counts.reindex(categories, fill_value=0), color='green', width=bar_width, label='Test')
        plt.bar(x + bar_width, validation_counts.reindex(categories, fill_value=0), color='purple', width=bar_width, label='Validation')

        # Adjust x-ticks for categorical variables
        plt.xticks(x, [int(val) for val in categories], rotation=45, fontsize=tick_font_size)
        # Add legend
        plt.legend(fontsize=legend_font_size)
        # Add labels and title
        plt.xlabel(f'({column}) [{column_units[column]}]', fontsize=label_font_size)
        plt.ylabel('Frequency [%]', fontsize=label_font_size)
        # Adjust tick parameters


        # Ensure layout is adjusted to prevent text from overlapping
        plt.tight_layout()



    else:
        train_column = train[column]
        test_column = test[column]
        validation_column = validation[column]

        sns.kdeplot(data=train_column, color='yellow', label='Train', shade=True)
        sns.kdeplot(data=test_column, color='green', label='Test', shade=True)
        sns.kdeplot(data=validation_column, color='purple', label='Validation', shade=True)

        # Add legend
        plt.legend(fontsize=legend_font_size)
        # Add labels and title
        plt.xlabel(f'{column} [{column_units[column]}]', fontsize=label_font_size)
        plt.ylabel('Density', fontsize=label_font_size)
        # Adjust tick parameters
        if column == 'SP':
            plt.xticks(fontsize=small_tick_font_size)
        else:
            plt.xticks(fontsize=tick_font_size)
        plt.yticks(fontsize=tick_font_size)


        # Ensure layout is adjusted to prevent text from overlapping
        plt.tight_layout()

        # Show the plot (for debugging)
    plt.show()

    # Save the KDE plot
    save_path = os.path.join(save_folder_path, f'{column}_kde.png')
    print(f"Saving plot to: {save_path}")
    plt.savefig(save_path)

    # Close the plot to prevent overlapping when creating the next one
    plt.close()

























#for column in columns:
#    plt.figure()
#    sns.kdeplot(data=train[column], color='yellow', label='Train', shade=True)
#    sns.kdeplot(data=test[column], color='green', label='Test', shade=True)
#    sns.kdeplot(data=validation[column], color='purple', label='Validation', shade=True)

#    # Add legend
#    plt.legend(fontsize=legend_font_size)

#    # Add labels and title
#    plt.xlabel(f'{column} [{column_units[column]}]', fontsize=label_font_size)
#    plt.ylabel('Density', fontsize=label_font_size)
#    # Adjust tick parameters
#    plt.xticks(fontsize=tick_font_size)
#    plt.yticks(fontsize=tick_font_size)
#    # Ensure layout is adjusted to prevent text from overlapping
#    plt.tight_layout()

#    # Show the plot (for debugging)
#    plt.show()

#    # Save the KDE plot
#    save_path = os.path.join(save_folder_path, f'{column}_kde.png')
#    print(f"Saving plot to: {save_path}")
#    plt.savefig(save_path)
    
#    # Close the plot to prevent overlapping when creating the next one
#    plt.close()




