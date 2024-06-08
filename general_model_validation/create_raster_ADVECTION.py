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


city='Brussels'
months='07'
years=2017
hours=23
days=25


all_data=[]
# Load the data
file_path = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_test/CITIES3/ready_for_model_{city}_{years}_{months}.csv'
all_data_combined = pd.read_csv(file_path)







#----------------------------------------------------------------------------------------------------------------------
#FOR ALL and ALL
city_data=all_data_combined[all_data_combined['hour']==hours]
city_data=city_data[city_data['day']==days]

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
model = joblib.load("/kyukon/data/gent/vo/000/gvo00041/vsc46127/FEATURESEL/model_FINAL.joblib")

# Predict
city_data['y_pred'] = model.predict(X_test)+city_data['T2M']
city_data['y_test'] =city_data['T2M_difference']+city_data['T2M']



# Calculate Bias
bias= np.mean(city_data['y_pred'] - city_data['y_test'])
# Calculate RMSE
rmse= np.sqrt(np.mean((city_data['y_pred'] - city_data['y_test'])**2))



results = city_data[['x', 'y', 'year', 'month', 'hour', 'y_test', 'T2M', 'y_pred', 'WS']]

# Convert columns to float
#results['y_pred'] = results['y_pred'].astype(float)
results['y_test'] = results['y_test'].astype(float)
#results['SOLAR_ELEV'] = results['SOLAR_ELEV'].astype(float)


# Reset the index of the DataFrame
results_reset_index = results.reset_index()

# Set a unique index for the DataFrame
results = results_reset_index.set_index(['year', 'month', 'hour', 'x', 'y'])



# Convert the DataFrame to an xarray dataset
xarray_dataset = results.to_xarray()

# Set CRS
celcius='\u00B0C'
degree='\u00B0'
xarray_dataset = xarray_dataset.rio.set_crs('EPSG:4326')
xarray_dataset[f'T2M_UrbClim[{celcius}]'] = xarray_dataset['y_test']
xarray_dataset[f'T2M_RFmodel[{celcius}]'] = xarray_dataset['y_pred']
xarray_dataset[f'T2M_diff[{celcius}]']=xarray_dataset['y_pred']-xarray_dataset['y_test']


wind_speed=np.mean(xarray_dataset['WS'])
print(type(wind_speed))
print(wind_speed)
wind_speed_value = float(wind_speed.values)

# Format wind speed value to two decimal places
wind_speed = "{:.2f}".format(wind_speed_value)

cmap = 'YlOrRd'



# Print the resulting DataFrame with mean 'y_pred' values for each 'x' and 'y'
# Compute common min and max values
vmin = min(xarray_dataset[f'T2M_RFmodel[{celcius}]'].min(), xarray_dataset[f'T2M_UrbClim[{celcius}]'].min())-0.5
vmax = max(xarray_dataset[f'T2M_RFmodel[{celcius}]'].max(), xarray_dataset[f'T2M_UrbClim[{celcius}]'].max())+0.5

# Plot 'y_pred' for the first timestep
plt.figure(figsize=(10, 8))
mappable_pred =xarray_dataset[f'T2M_UrbClim[{celcius}]'].plot(x='x', y='y', cmap=cmap, vmin=vmin, vmax=vmax, label=f'Predicted T2M [{celcius}]')
plt.title(f'UrbClim T2M plot: {str(days)}/{str(months)}/{str(years)} {hours}h:00 \n{city}\nWind speed: {wind_speed}m/s', fontsize=16)

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
plt.tight_layout()
plt.savefig(f'UrbClim_T2m_{city}_{years}_{days}_{months}_{hours}.png', format='png')
plt.close()




# Plot 'y_pred' for the first timestep
plt.figure(figsize=(10, 8))
mappable_pred = xarray_dataset[f'T2M_RFmodel[{celcius}]'].plot(x='x', y='y', cmap=cmap, vmin=vmin, vmax=vmax, label=f'Predicted T2M [{celcius}]')
plt.title(f'RFmodel T2M plot: {str(days)}/{str(months)}/{str(years)} {hours}h:00 \n{city}\nWind speed: {wind_speed}m/s\nBias: {bias:.4f} {celcius}, RMSE: {rmse:.4f} {celcius} :', fontsize=16)

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
plt.tight_layout()
plt.savefig(f'RFmodel_T2m_{city}_{years}_{days}_{months}_{hours}.png', format='png')
plt.close()











#plot difference
# Calculate the difference


# Plot 'y_pred - y_test' for the first timestep with title 'Difference'
plt.figure(figsize=(10, 8))
mappable_diff = xarray_dataset[f'T2M_diff[{celcius}]'].plot(x='x', y='y', cmap='seismic', vmin=-2, vmax=2)
plt.title(f'Difference T2m (RFmodel - UrbClim): {str(days)}/{str(months)}/{str(years)} {hours}h:00 \n{city}\nBias: {bias:.4f} {celcius}, RMSE: {rmse:.4f} {celcius}', fontsize=18)
plt.xlabel('Longitude', fontsize=18)
plt.ylabel('Latitude', fontsize=18)
x_ticks = np.linspace(plt.xticks()[0][0], plt.xticks()[0][-1], num=6)
y_ticks = np.linspace(plt.yticks()[0][0], plt.yticks()[0][-1], num=6)

x_direction = ['E' if tick > 0 else 'W' for tick in x_ticks]
y_direction = ['N' if tick > 0 else 'S' for tick in y_ticks]

x_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(x_ticks, x_direction)]
y_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(y_ticks, y_direction)]
# Set the modified tick labels with increased font size
plt.xticks(x_ticks, labels=x_tick_labels, fontsize=17)
plt.yticks(y_ticks, labels=y_tick_labels, fontsize=17)
mappable_diff.colorbar.ax.tick_params(labelsize=18)
mappable_diff.colorbar.set_label(f'T2M Difference [{celcius}]', size=18)
# Set the title of the colorbar
# Create a colorbar for the difference plot and save the plot
plt.savefig(f'difference_plot_{city}_{years}_{days}_{months}_{hours}.png', format='png')
plt.close()




