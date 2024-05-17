#import xarray as xr
#import rioxarray as rxr
#import os
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_squared_error
import joblib
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import codecs
celcius='\u00B0C'







file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/test_data_FINAL.csv'
test = pd.read_csv(file_path, usecols=['T_2M', 'city','T_TARGET','T_2M_COR', 'LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL', 'ruralurbanmask', 'city', 'hour'])

cluster_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/cluster/cities_test_FINAL.csv'
cluster_data = pd.read_csv(cluster_path)
cluster_data['Cluster']=cluster_data['Cluster']+1
cluster_data = cluster_data.rename(columns={'City': 'city'})
# Add a new column containing the first 4 letters of the city name
test = pd.merge(cluster_data, test, on='city')


# Count NaN values per column before dropping
test_nan_before_drop = test.isnull().sum()

# Drop rows with NaN values in 'T_TARGET' column
test = test.dropna(subset=['T_TARGET'])

test = test.rename(columns={'T_2M': 'T2M_NC'})
test = test.rename(columns={'T_2M_COR': 'T2M'})

test['T2M_difference'] = test['T_TARGET'] - test['T2M']
test['T2M'] = test['T2M'] - 273.15
test['T2M_NC'] = test['T2M_NC'] - 273.15

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


end_time = time.perf_counter()
elapsed_time = end_time - start_time

# Write results to file
with open('run_time.txt', 'a') as f:
    f.write(f"Elapsed time impute:  {elapsed_time} seconds  \n")




# Count NaN values per column after filling with median
test_nan_after_median_fill = test.isnull().sum()
# Calculate number of NaN values filled and dropped per column
test_nan_filled = test_nan_before_drop - test_nan_after_fill
test_nan_dropped = test_nan_after_drop - test_nan_after_median_fill
# Writing results to a file
with open('run_time.txt', 'a') as f:
    f.write("NaN values filled per column before dropping 'T_TARGET':\n")
    f.write("Test Data:\n")
    f.write(test_nan_filled.to_string() + "\n")
    f.write("\n")
    f.write("NaN values dropped per column after dropping 'T_TARGET':\n")
    f.write("Test Data:\n")
    f.write(test_nan_dropped.to_string() + "\n")

model = joblib.load("/kyukon/data/gent/vo/000/gvo00041/vsc46127/FEATURESEL/model_FINAL.joblib")

test['y_test']= test['T2M_difference']+test['T2M']
X_test = test[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
test['y_pred'] = model.predict(X_test)+test['T2M']


# Load the model



def calculate_bias(y_true, y_pred):
    return ((y_pred - y_true).mean())


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



import matplotlib.pyplot as plt
import seaborn as sns

# Filter for urban and rural data
urban = test[test['ruralurbanmask'] == 0]
rural = test[test['ruralurbanmask'] == 1]


# Function to calculate bias
def calculate_bias(y_true, y_pred):
    return (np.mean(y_pred - y_true))


# Plotting setup for Bias

# Set color palette for each cluster
#sns.set_palette("husl", len(urban['Cluster'].unique()))



# Iterate over clusters
for Cluster in urban['Cluster'].unique():
    start_time = time.perf_counter()

    cluster_df = urban[urban['Cluster'] == Cluster]
    hourly_bias_model = cluster_df.groupby('hour').apply(lambda group: calculate_bias(group['y_test'], group['y_pred']))
    hourly_bias_ERA5 = cluster_df.groupby('hour').apply(lambda group: calculate_bias(group['y_test'], group['T2M']))
    # Write results to file

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    with open('run_time.txt', 'a') as f:
        f.write(f" hourly_bias_model { hourly_bias_model}  \n")

    with open('run_time.txt', 'a') as f:
        f.write(f"hourly_bias_ERA5 {hourly_bias_ERA5}  \n")

    # Example: Create a custom x-axis label for each hour
    hour_labels = [f"{int(hour):02d}:00" for hour in hourly_bias_model.index]
    print(hourly_bias_model.head())  # Check the first few rows of the DataFrame
    plt.figure(figsize=(12, 8))

    # Plot hourly biases for model
    plt.plot(hourly_bias_model.index, hourly_bias_model.values,color='orange',label=f'RFmodel')
    # Plot hourly biases for ERA5
    plt.plot(hourly_bias_ERA5.index, hourly_bias_ERA5.values, color='b',label=f'ERA5')


    # Customize plot for Bias
    plt.title(f'Diurnal urban bias plot: Cluster {Cluster}', fontsize= 23)
    plt.ylabel(f'Bias [{celcius}]', fontsize=22)
    #plt.xlabel('Hour')
    plt.xticks(hourly_bias_model.index, hour_labels, rotation='vertical', fontsize=21)  # Set custom x-axis labels
    plt.yticks(fontsize=21)  # Adjust fontsize for y-tick labels
    plt.ylim(-2.5, 1)

    plt.grid(True)
    plt.legend(loc='upper right', fontsize=22)
    plt.tight_layout()

    # Save plot for urban bias
    plot_filename_bias_urban = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/urban_hourly_bias_plot_Cluster{Cluster}.png'
    plt.savefig(plot_filename_bias_urban, format='png')




# Iterate over clusters
for Cluster in rural['Cluster'].unique():
    start_time = time.perf_counter()

    cluster_df = rural[rural['Cluster'] == Cluster]
    hourly_bias_model = cluster_df.groupby('hour').apply(lambda group: calculate_bias(group['y_test'], group['y_pred']))
    hourly_bias_ERA5 = cluster_df.groupby('hour').apply(lambda group: calculate_bias(group['y_test'], group['T2M']))
    # Write results to file

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    with open('run_time.txt', 'a') as f:
        f.write(f" hourly_bias_model { hourly_bias_model}  \n")

    with open('run_time.txt', 'a') as f:
        f.write(f"hourly_bias_ERA5 {hourly_bias_ERA5}  \n")

    # Example: Create a custom x-axis label for each hour
    hour_labels = [f"{int(hour):02d}:00" for hour in hourly_bias_model.index]
    print(hourly_bias_model.head())  # Check the first few rows of the DataFrame
    plt.figure(figsize=(12, 8))

    # Plot hourly biases for model
    plt.plot(hourly_bias_model.index, hourly_bias_model.values,color='orange',label=f'RFmodel')
    # Plot hourly biases for ERA5
    plt.plot(hourly_bias_ERA5.index, hourly_bias_ERA5.values, color='b',label=f'ERA5')


    # Customize plot for Bias
    plt.title(f'Diurnal rural bias plot: Cluster {Cluster}', fontsize= 23)
    plt.ylabel(f'Bias [{celcius}]', fontsize=22)
    #plt.xlabel('Hour')
    plt.xticks(hourly_bias_model.index, hour_labels, rotation='vertical', fontsize=21)  # Set custom x-axis labels
    plt.yticks(fontsize=21)  # Adjust fontsize for y-tick labels
    plt.ylim(-2.5, 1)

    plt.grid(True)
    plt.legend(loc='upper right', fontsize=22)
    plt.tight_layout()

    # Save plot for rural bias
    plot_filename_bias_rural = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/rural_hourly_bias_plot_Cluster{Cluster}.png'
    plt.savefig(plot_filename_bias_rural, format='png')

