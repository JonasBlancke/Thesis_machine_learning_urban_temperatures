import xarray as xr
import pandas as pd
import numpy as np
import joblib
import rasterio
import rioxarray as rxr
import matplotlib.pyplot as plt
import codecs
celcius='\u00B0C'
degree='\u00B0'


city='Marseille'
all_months=['01', '02', '03','04','05','06', '07', '08', '09', '10', '11', '12']
years=[2015, 2016, 2017]



all_data=[]
for year in years:
    for month in all_months:
        # Load the data
        file_path = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_test/MODEL_PART1_TEST_NO_SPSUB/ready_for_model_{city}_{year}_{month}.csv'
        city_data = pd.read_csv(file_path)
        # Append city_data to summer_data list
        all_data.append(city_data)
        # Write results to file
        with open('Marseille.txt', 'a') as f:
            f.write(f"Succesfull for {city} {year} {month}  \n")


# Concatenate all DataFrames in summer_data list
all_data_combined = pd.concat(all_data, ignore_index=True)








#----------------------------------------------------------------------------------------------------------------------
#FOR ALL and ALL
city_data=all_data_combined
print(city_data.columns)


# Drop rows with NaN values in 'T_TARGET' column
city_data = city_data.dropna(subset=['T_TARGET'])

city_data = city_data.rename(columns={'T_2M': 'T2M_NC'})
city_data = city_data.rename(columns={'T_2M_COR': 'T2M', 'T_SK': 'SKT'})

city_data['T2M_difference'] = city_data['T_TARGET'] - city_data['T2M']
city_data['T2M'] = city_data['T2M'] - 273.15
city_data['T_TARGET'] = city_data['T_TARGET'] - 273.15
city_data['SKT'] = city_data['SKT'] - 273.15
city_data['T2M_NC'] = city_data['T2M_NC'] - 273.15

city_data = city_data.dropna(subset=['T2M_difference'])

for column in city_data.columns:
    if city_data[column].isnull().any():
        city_data[column] = city_data[column].transform(lambda x: x.fillna(x.median()))





# Select features and target
X_test = city_data[['LC_CORINE', 'IMPERV', 'HEIGHT', 'COAST', 'ELEV', 'POP', 'RH', 'SP', 'PRECIP','T2M', 'WS', 'TCC', 'CAPE', 'BLH', 'SSR', 'SOLAR_ELEV', 'DECL']]
y_test = city_data['T2M_difference']


# Load the trained model
#model = joblib.load("/kyukon/data/gent/vo/000/gvo00041/vsc46127/FEATURESEL/model_FINAL.joblib")

# Predict
#y_pred = model.predict(X_test)


city_data['y_test'] =city_data['T_TARGET']


city_data_filter=city_data
# Assuming 'y_pred' and 'y_test' are Pandas Series in city_data
y_pred = city_data_filter['T2M_NC']
y_test = city_data_filter['T_TARGET']

# Calculate Bias
bias_NC = np.mean(y_pred - y_test)
# Calculate RMSE
rmse_NC = np.sqrt(np.mean((y_pred - y_test)**2))

y_pred = city_data_filter['T2M']
y_test = city_data_filter['T_TARGET']

# Calculate Bias
bias_C = np.mean(y_pred - y_test)
# Calculate RMSE
rmse_C = np.sqrt(np.mean((y_pred - y_test)**2))


results = city_data_filter[['x', 'y', 'year', 'month', 'hour', 'y_test', 'SOLAR_ELEV', 'T2M','T2M_NC']]

# Convert columns to float
#results['y_pred'] = results['y_pred'].astype(float)
results['y_test'] = results['y_test'].astype(float)
#results['SOLAR_ELEV'] = results['SOLAR_ELEV'].astype(float)


# Reset the index of the DataFrame
results_reset_index = results.reset_index()

# Set a unique index for the DataFrame
results = results_reset_index.set_index(['year', 'month', 'hour', 'x', 'y'])

mean_results = results.groupby(['x', 'y']).mean()


# Convert the DataFrame to an xarray dataset
xarray_dataset = mean_results.to_xarray()

# Set CRS
celcius='\u00B0C'
degree='\u00B0'
mean_xarray_dataset = xarray_dataset.rio.set_crs('EPSG:4326')
mean_xarray_dataset[f'T2M_UrbClim[{celcius}]'] = mean_xarray_dataset['y_test']
mean_xarray_dataset[f'Lapse_rate_correction[{celcius}]'] = mean_xarray_dataset['T2M']-mean_xarray_dataset['T2M_NC']



# Print the xarray dataset structure
print(mean_xarray_dataset)

cmap = 'YlOrRd'

# Calculate the mean 'y_pred' value for each unique combination of 'x' and 'y'

print(mean_xarray_dataset)


# Print the resulting DataFrame with mean 'y_pred' values for each 'x' and 'y'
# Compute common min and max values
vmin = min(mean_xarray_dataset[f'T2M_UrbClim[{celcius}]'].min(), mean_xarray_dataset[f'T2M_UrbClim[{celcius}]'].min())-0.5
vmax = max(mean_xarray_dataset['T2M_NC'].max(), mean_xarray_dataset['T2M_NC'].max())+0.5

# Plot 'y_pred' for the first timestep
plt.figure(figsize=(10, 8))
mappable_pred = mean_xarray_dataset['T2M_NC'].plot(x='x', y='y', cmap=cmap, vmin=vmin, vmax=vmax, label=f'Predicted T2M [{celcius}]')
plt.title(f'Average non corrected ERA5 temperature (2015-2017)\n{city}\nBias: {bias_NC:.4f} {celcius}, RMSE: {rmse_NC:.4f} {celcius} :', fontsize=16)

plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
# Adding direction indicators for x and y axes
x_ticks = plt.xticks()[0]
y_ticks = plt.yticks()[0]

x_direction = ['E' if tick > 0 else 'W' for tick in x_ticks]
y_direction = ['N' if tick > 0 else 'S' for tick in y_ticks]

x_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(x_ticks, x_direction)]
y_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(y_ticks, y_direction)]
# Set the modified tick labels with increased font size
plt.xticks(x_ticks, labels=x_tick_labels, fontsize=13)
plt.yticks(y_ticks, labels=y_tick_labels, fontsize=13)
mappable_pred.colorbar.ax.tick_params(labelsize=16)
mappable_pred.colorbar.set_label(f'T2M [{celcius}]', size=16)
plt.savefig(f'NONCORRECTED_T2m_{city}_average_plot_2015_2017.png', format='png')
plt.close()




# Plot 'y_pred' for the first timestep
plt.figure(figsize=(10, 8))
mappable_pred = mean_xarray_dataset['T2M'].plot(x='x', y='y', cmap=cmap, vmin=vmin, vmax=vmax, label=f'Predicted T2M [{celcius}]')
plt.title(f'Average lapse rate corrected ERA5 temperature (2015-2017)\n{city}\nBias: {bias_C:.4f} {celcius}, RMSE: {rmse_C:.4f} {celcius} :', fontsize=16)

plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
# Adding direction indicators for x and y axes
x_ticks = plt.xticks()[0]
y_ticks = plt.yticks()[0]

x_direction = ['E' if tick > 0 else 'W' for tick in x_ticks]
y_direction = ['N' if tick > 0 else 'S' for tick in y_ticks]

x_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(x_ticks, x_direction)]
y_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(y_ticks, y_direction)]
# Set the modified tick labels with increased font size
plt.xticks(x_ticks, labels=x_tick_labels, fontsize=13)
plt.yticks(y_ticks, labels=y_tick_labels, fontsize=13)
mappable_pred.colorbar.ax.tick_params(labelsize=16)
mappable_pred.colorbar.set_label(f'T2M [{celcius}]', size=16)
plt.savefig(f'CORRECTED_T2m_{city}_average_plot_2015_2017.png', format='png')
plt.close()










# Plot 'y_test' for the first timestep with title 'UrbClim'
plt.figure(figsize=(10, 8))
mappable_test = mean_xarray_dataset[f'T2M_UrbClim[{celcius}]'].plot(x='x', y='y', cmap=cmap, vmin=vmin, vmax=vmax, label=f'UrbClim T2M [{celcius}]')
plt.title(f'Average UrbClim T2m (2015-2017)\n{city}\n ', fontsize=16)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
x_ticks = plt.xticks()[0]
y_ticks = plt.yticks()[0]

x_direction = ['E' if tick > 0 else 'W' for tick in x_ticks]
y_direction = ['N' if tick > 0 else 'S' for tick in y_ticks]

x_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(x_ticks, x_direction)]
y_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(y_ticks, y_direction)]
# Set the modified tick labels with increased font size
plt.xticks(x_ticks, labels=x_tick_labels, fontsize=13)
plt.yticks(y_ticks, labels=y_tick_labels, fontsize=13)

mappable_test.colorbar.ax.tick_params(labelsize=16)
mappable_test.colorbar.set_label(f'T2M [{celcius}]', size=16)
plt.savefig(f'UrbClim_T2m_{city}_average_plot_in_2015_2017.png', format='png')
plt.close()







#plot lapse rate
# Calculate the difference
vmin = min(mean_xarray_dataset[f'Lapse_rate_correction[{celcius}]'].min(), mean_xarray_dataset[f'Lapse_rate_correction[{celcius}]'].min())-0.5
vmax = max(mean_xarray_dataset[f'Lapse_rate_correction[{celcius}]'].max(), mean_xarray_dataset[f'Lapse_rate_correction[{celcius}]'].max())+0.5


plt.figure(figsize=(10, 8))
mappable_diff = mean_xarray_dataset[f'Lapse_rate_correction[{celcius}]'].plot(x='x', y='y', cmap='seismic', vmin=vmin, vmax=vmax)
plt.title(f'Lapse rate correction \n{city}\n', fontsize=16)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
x_ticks = plt.xticks()[0]
y_ticks = plt.yticks()[0]

x_direction = ['E' if tick > 0 else 'W' for tick in x_ticks]
y_direction = ['N' if tick > 0 else 'S' for tick in y_ticks]

x_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(x_ticks, x_direction)]
y_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(y_ticks, y_direction)]
# Set the modified tick labels with increased font size
plt.xticks(x_ticks, labels=x_tick_labels, fontsize=13)
plt.yticks(y_ticks, labels=y_tick_labels, fontsize=13)
mappable_diff.colorbar.ax.tick_params(labelsize=16)
mappable_diff.colorbar.set_label(f'T2M [{celcius}]', size=16)
# Set the title of the colorbar
# Create a colorbar for the difference plot and save the plot
plt.savefig(f'Average lapseratecorrection_plot_{city}_plot.png', format='png')
plt.close()





