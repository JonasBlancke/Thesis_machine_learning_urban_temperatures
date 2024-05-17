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

import pandas as pd
import matplotlib.pyplot as plt

# Plotting RMSE vs Max Depth
# Load the dataframes
df1 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_depth_1.csv")
df2 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_depth_2.csv")
df3 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_depth_3.csv")

# Create a new figure
plt.figure(figsize=(10, 8))  # Increased figure size

# Plotting
plt.plot(df1['max_depth'], df1['rmse_val'], label='Validation default 1', color='tab:blue', linestyle='--', linewidth=2)  # Increased line thickness
plt.plot(df1['max_depth'], df1['rmse_train'], label='Training default 1', color='tab:blue', linewidth=2)  # Increased line thickness
plt.plot(df2['max_depth'], df2['rmse_val'], label='Validation default 2', color='tab:orange', linestyle='--', linewidth=2)  # Increased line thickness
plt.plot(df2['max_depth'], df2['rmse_train'], label='Training default 2', color='tab:orange', linewidth=2)  # Increased line thickness
plt.plot(df3['max_depth'], df3['rmse_val'], label='Validation default 3', color='tab:green', linestyle='--', linewidth=2)  # Increased line thickness
plt.plot(df3['max_depth'], df3['rmse_train'], label='Training default 3', color='tab:green', linewidth=2)  # Increased line thickness

plt.xlabel('Max Depth', fontsize=17)  # Increased font size
plt.ylabel('RMSE', fontsize=17)  # Increased font size
plt.legend(loc='upper right', fontsize=15)  # Adjusted legend position and increased font size
plt.grid(True)

# Save the plot
plt.savefig('rmse_vs_max_depth_plot.png')

plt.show()


# Plotting RMSE vs n_estimators
# Load the dataframes
df1 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_n_estimators_1.csv")
df2 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_n_estimators_2.csv")
df3 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_n_estimators_3.csv")

# Create a new figure
plt.figure(figsize=(10, 8))  # Increased figure size

# Plotting
plt.plot(df1['n_estimators'], df1['rmse_val'], label='Validation default 1', color='tab:blue', linestyle='--', linewidth=2)  # Increased line thickness
plt.plot(df1['n_estimators'], df1['rmse_train'], label='Training default 1', color='tab:blue', linewidth=2)  # Increased line thickness
plt.plot(df2['n_estimators'], df2['rmse_val'], label='Validation default 2', color='tab:orange', linestyle='--', linewidth=2)  # Increased line thickness
plt.plot(df2['n_estimators'], df2['rmse_train'], label='Training default 2', color='tab:orange', linewidth=2)  # Increased line thickness
plt.plot(df3['n_estimators'], df3['rmse_val'], label='Validation default 3', color='tab:green', linestyle='--', linewidth=2)  # Increased line thickness
plt.plot(df3['n_estimators'], df3['rmse_train'], label='Training default 3', color='tab:green', linewidth=2)  # Increased line thickness

plt.xlabel('n estimators', fontsize=17)  # Increased font size
plt.ylabel('RMSE', fontsize=17)  # Increased font size
plt.legend(loc='upper right', fontsize=15)  # Adjusted legend position and increased font size
plt.grid(True)

# Save the plot
plt.savefig('rmse_vs_n_estimators_plot.png')

plt.show()


# Plotting RMSE vs max_features
# Load the dataframes
df1 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_features_1.csv")
df2 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_features_2.csv")
df3 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_features_3.csv")

# Create a new figure
plt.figure(figsize=(10, 8))  # Increased figure size

# Plotting
plt.plot(df1['max_features'], df1['rmse_val'], label='Validation default 1', color='tab:blue', linestyle='--', linewidth=2)  # Increased line thickness
plt.plot(df1['max_features'], df1['rmse_train'], label='Training default 1', color='tab:blue', linewidth=2)  # Increased line thickness
plt.plot(df2['max_features'], df2['rmse_val'], label='Validation default 2', color='tab:orange', linestyle='--', linewidth=2)  # Increased line thickness
plt.plot(df2['max_features'], df2['rmse_train'], label='Training default 2', color='tab:orange', linewidth=2)  # Increased line thickness
plt.plot(df3['max_features'], df3['rmse_val'], label='Validation default 3', color='tab:green', linestyle='--', linewidth=2)  # Increased line thickness
plt.plot(df3['max_features'], df3['rmse_train'], label='Training default 3', color='tab:green', linewidth=2)  # Increased line thickness

plt.xlabel('max_features', fontsize=17)  # Increased font size
plt.ylabel('RMSE', fontsize=17)  # Increased font size
plt.legend(loc='upper right', fontsize=15)  # Adjusted legend position and increased font size
plt.grid(True)

# Save the plot
plt.savefig('rmse_vs_max_features_plot.png')

plt.show()

# Close all plots
plt.close('all')


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframes
df_depth_1 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_depth_1.csv")
df_depth_2 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_depth_2.csv")
df_depth_3 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_depth_3.csv")

df_estimators_1 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_n_estimators_1.csv")
df_estimators_2 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_n_estimators_2.csv")
df_estimators_3 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_n_estimators_3.csv")

df_features_1 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_features_1.csv")
df_features_2 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_features_2.csv")
df_features_3 = pd.read_csv("/kyukon/data/gent/vo/000/gvo00041/vsc46127/results_max_features_3.csv")

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(24, 8))  # 3 subplots horizontally arranged

# Plotting for max_depth
axs[0].plot(df_depth_1['max_depth'], df_depth_1['rmse_val'], label='Validation 1', color='tab:blue', linestyle='--', linewidth=2)
axs[0].plot(df_depth_1['max_depth'], df_depth_1['rmse_train'], label='Train 1', color='tab:blue', linewidth=2)
axs[0].plot(df_depth_2['max_depth'], df_depth_2['rmse_val'], label='Validation 2', color='tab:orange', linestyle='--', linewidth=2)
axs[0].plot(df_depth_2['max_depth'], df_depth_2['rmse_train'], label='Train 2', color='tab:orange', linewidth=2)
axs[0].plot(df_depth_3['max_depth'], df_depth_3['rmse_val'], label='Validation 3', color='tab:green', linestyle='--', linewidth=2)
axs[0].plot(df_depth_3['max_depth'], df_depth_3['rmse_train'], label='Train 3', color='tab:green', linewidth=2)
axs[0].set_xlabel('Max Depth', fontsize=20)
axs[0].set_ylabel('RMSE', fontsize=20)
axs[0].legend(loc='upper right', fontsize=16)
axs[0].grid(True)
axs[0].tick_params(axis='both', which='major', labelsize=15)  # Increase tick label size

# Plotting for n_estimators
axs[1].plot(df_estimators_1['n_estimators'], df_estimators_1['rmse_val'], label='Validation 1', color='tab:blue', linestyle='--', linewidth=2)
axs[1].plot(df_estimators_1['n_estimators'], df_estimators_1['rmse_train'], label='Train 1', color='tab:blue', linewidth=2)
axs[1].plot(df_estimators_2['n_estimators'], df_estimators_2['rmse_val'], label='Validation 2', color='tab:orange', linestyle='--', linewidth=2)
axs[1].plot(df_estimators_2['n_estimators'], df_estimators_2['rmse_train'], label='Train 2', color='tab:orange', linewidth=2)
axs[1].plot(df_estimators_3['n_estimators'], df_estimators_3['rmse_val'], label='Validation 3', color='tab:green', linestyle='--', linewidth=2)
axs[1].plot(df_estimators_3['n_estimators'], df_estimators_3['rmse_train'], label='Train 3', color='tab:green', linewidth=2)
axs[1].set_xlabel('n_estimators', fontsize=20)
axs[1].set_ylabel('RMSE', fontsize=20)
axs[1].legend(loc='upper right', fontsize=16)
axs[1].grid(True)
axs[1].tick_params(axis='both', which='major', labelsize=15)  # Increase tick label size

# Plotting for max_features
axs[2].plot(df_features_1['max_features'], df_features_1['rmse_val'], label='Validation 1', color='tab:blue', linestyle='--', linewidth=2)
axs[2].plot(df_features_1['max_features'], df_features_1['rmse_train'], label='Train 1', color='tab:blue', linewidth=2)
axs[2].plot(df_features_2['max_features'], df_features_2['rmse_val'], label='Validation 2', color='tab:orange', linestyle='--', linewidth=2)
axs[2].plot(df_features_2['max_features'], df_features_2['rmse_train'], label='Train 2', color='tab:orange', linewidth=2)
axs[2].plot(df_features_3['max_features'], df_features_3['rmse_val'], label='Validation 3', color='tab:green', linestyle='--', linewidth=2)
axs[2].plot(df_features_3['max_features'], df_features_3['rmse_train'], label='Train 3', color='tab:green', linewidth=2)
axs[2].set_xlabel('max_features', fontsize=20)
axs[2].set_ylabel('RMSE', fontsize=20)
axs[2].legend(loc='upper right', fontsize=16)
axs[2].grid(True)
axs[2].tick_params(axis='both', which='major', labelsize=15)  # Increase tick label size

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('rmse_subplots_horizontal_with_ticks.png')

plt.show()
