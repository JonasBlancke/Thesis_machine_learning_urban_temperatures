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
import codecs
celcius='\u00B0C'
degree='\u00B0'



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
    14: 'Inland water bodies',
    15: 'Sea'
}

file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/test_data_FINAL.csv'
val = pd.read_csv(file_path, usecols=['T_TARGET','LC_CORINE', 'city', 'T_2M_COR'])
val = val.rename(columns={'T_2M_COR': 'T2M'})

val['T2M_difference'] = val['T_TARGET'] - val['T2M']



# Count NaN values per column before dropping
val_nan_before_drop = val.isnull().sum()

# Drop rows with NaN values in 'T_TARGET' column
val = val.dropna(subset=['T_TARGET'])

# Drop rows with NaN values in 'T_TARGET' column
val = val.dropna(subset=['T_TARGET'])

import seaborn as sns
import matplotlib.pyplot as plt


#LC CORINE: 1, 2, 3, ..., 14

# Create boxplot for all cities
sns.boxplot(data=val, x='LC_CORINE', y='T2M_difference', showfliers=False)
plt.xticks(ticks=range(0, 15), labels=[label_mapping[x] for x in range(1, 16)], rotation=90, ha='center')
plt.ylabel(f'ERA5 residual [{celcius}]', fontsize=12)
plt.tight_layout()
plt.savefig('boxplot_all_cities.png')
plt.show()
plt.close()

# Create boxplot for Frankfurt
Frankfurt = val[val['city'] == "Frankfurt_am_Main"]
sns.boxplot(data=Frankfurt, x='LC_CORINE', y='T2M_difference', showfliers=False)
plt.xticks(ticks=range(0, 14), labels=[label_mapping[x] for x in range(1, 15)], rotation=90, ha='center')
plt.ylabel(f'ERA5 residual [{celcius}]', fontsize=12)
plt.tight_layout()
plt.savefig('boxplot_frankfurt.png')
plt.show()

