import xarray as xr
import pandas as pd
import numpy as np
import joblib
import rasterio
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors  # Add this import statement
import codecs
celcius='\u00B0C'
degree='\u00B0'


city='Frankfurt_am_Main'
months=['01', '11']
years=[2015, 2017]




data = []


for year in years:
    for month in months:
        # Load the data
        file_path = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_test/MODEL_PART1_TEST_NO_SPSUB/ready_for_model_{city}_{year}_{month}.csv'
        city_data = pd.read_csv(file_path)
        # Append city_data to summer_data list
        data.append(city_data)
        # Write results to file
        with open('Marseille.txt', 'a') as f:
            f.write(f"Succesfull for {city} {year} {month}  \n")


# Concatenate all DataFrames in summer_data list
data_combined = pd.concat(data, ignore_index=True)








#----------------------------------------------------------------------------------------------------------------------
#FOR ALL
city_data=data_combined
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
y_pred = model.predict(X_test)
city_data['y_pred'] = y_pred + city_data['T2M']
city_data['y_test'] =city_data['T_TARGET']
city_data['ypred_target']=y_pred
city_data['ytest_target']=y_test





# Assuming 'y_pred' and 'y_test' are Pandas Series in city_data
y_pred = city_data['y_pred']
y_test = city_data['T_TARGET']

# Calculate Bias
bias = np.mean(y_pred - y_test)
# Calculate RMSE
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print(f'Bias: {bias:.4f} {celcius}')
print(f'RMSE : {rmse:.4f} {celcius}')

results = city_data[['x', 'y', 'year', 'month', 'hour', 'y_pred', 'y_test', 'SOLAR_ELEV', 'ypred_target', 'ytest_target', 'LC_CORINE', 'LCZ']]
average_ytest_target = city_data.groupby('LCZ')['ytest_target'].mean().reset_index()
# Rename the columns for clarity
average_ytest_target.columns = ['LCZ', 'Average_ytest_target']
# Export the result to a CSV file
average_ytest_target.to_csv(f'average_ytest_target_per_LC_CORINE_{city}.csv', index=False)

# Filter the data for day and night based on SOLAR_ELEV
day_data = city_data[city_data['SOLAR_ELEV'] > 0]
night_data = city_data[city_data['SOLAR_ELEV'] <= 0]

# Compute the average ytest_target for each LC_CORINE for day
average_ytest_target_day = day_data.groupby('LCZ')['ytest_target'].mean().reset_index()
# Rename the columns for clarity
average_ytest_target_day.columns = ['LCZ', 'Average_ytest_target']
# Export the result to a CSV file for day
average_ytest_target_day.to_csv(f'average_ytest_target_per_LC_CORINE_day_{city}.csv', index=False)

# Compute the average ytest_target for each LC_CORINE for night
average_ytest_target_night = night_data.groupby('LCZ')['ytest_target'].mean().reset_index()
# Rename the columns for clarity
average_ytest_target_night.columns = ['LCZ', 'Average_ytest_target']
# Export the result to a CSV file for night
average_ytest_target_night.to_csv(f'average_ytest_target_per_LC_CORINE_night_{city}.csv', index=False)



# Convert columns to float
results['y_pred'] = results['y_pred'].astype(float)
results['y_test'] = results['y_test'].astype(float)
results['ypred_target'] = results['ypred_target'].astype(float)
results['ytest_target'] = results['ytest_target'].astype(float)

results['SOLAR_ELEV'] = results['SOLAR_ELEV'].astype(float)


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
mean_xarray_dataset[f'T2M_pred[{celcius}]'] = mean_xarray_dataset['y_pred']
mean_xarray_dataset[f'T2M_UrbClim[{celcius}]'] = mean_xarray_dataset['y_test']
mean_xarray_dataset[f'T2M_diff[{celcius}]'] = mean_xarray_dataset['y_pred']-mean_xarray_dataset['y_test']

# Print the xarray dataset structure
print(mean_xarray_dataset)

cmap = 'YlOrRd'

# Calculate the mean 'y_pred' value for each unique combination of 'x' and 'y'

print(mean_xarray_dataset)


# Print the resulting DataFrame with mean 'y_pred' values for each 'x' and 'y'
# Compute common min and max values
vmin = min(mean_xarray_dataset[f'T2M_UrbClim[{celcius}]'].min(), mean_xarray_dataset[f'T2M_UrbClim[{celcius}]'].min())-0.1
vmax = max(mean_xarray_dataset[f'T2M_pred[{celcius}]'].max(), mean_xarray_dataset[f'T2M_UrbClim[{celcius}]'].max())+0.1

# Plot 'y_pred' for the first timestep
plt.figure(figsize=(10, 8))
mappable_pred = mean_xarray_dataset[f'T2M_pred[{celcius}]'].plot(x='x', y='y', cmap=cmap, vmin=vmin, vmax=vmax, label=f'Predicted T2M [{celcius}]')
plt.title(f'RFmodel_T2m: average plot \n{city} (2015-2017)\nBias: {bias:.4f} {celcius}, RMSE: {rmse:.4f} {celcius}', fontsize=18)

plt.xlabel('Longitude', fontsize=18)
plt.ylabel('Latitude', fontsize=18)
# Adding direction indicators for x and y axes
x_ticks = np.linspace(plt.xticks()[0][0], plt.xticks()[0][-1], num=6)
y_ticks = np.linspace(plt.yticks()[0][0], plt.yticks()[0][-1], num=6)


x_direction = ['E' if tick > 0 else 'W' for tick in x_ticks]
y_direction = ['N' if tick > 0 else 'S' for tick in y_ticks]

x_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(x_ticks, x_direction)]
y_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(y_ticks, y_direction)]
# Set the modified tick labels with increased font size
plt.xticks(x_ticks, labels=x_tick_labels, fontsize=17)
plt.yticks(y_ticks, labels=y_tick_labels, fontsize=17)
mappable_pred.colorbar.ax.tick_params(labelsize=18)
mappable_pred.colorbar.set_label(f'T2M RFmodel [{celcius}]', size=18)
plt.savefig(f'RFmodel_T2m_{city}_average_2015_2017.png', format='png')
plt.close()




# Plot 'y_test' for the first timestep with title 'UrbClim'
plt.figure(figsize=(10, 8))
mappable_test = mean_xarray_dataset[f'T2M_UrbClim[{celcius}]'].plot(x='x', y='y', cmap=cmap, vmin=vmin, vmax=vmax, label=f'UrbClim T2M [{celcius}]')
plt.title(f'UrbClim_T2m: average plot \n{city} (2015-2017)\n ' , fontsize=18)
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

mappable_test.colorbar.ax.tick_params(labelsize=18)
mappable_test.colorbar.set_label(f'T2M UrbClim [{celcius}]', size=18)
plt.savefig(f'UrbClim_T2m_{city}_average_2015_2017.png', format='png')
plt.close()




#plot difference
# Calculate the difference


# Plot 'y_pred - y_test' for the first timestep with title 'Difference'
plt.figure(figsize=(10, 8))
mappable_diff = mean_xarray_dataset[f'T2M_diff[{celcius}]'].plot(x='x', y='y', cmap='seismic', vmin=-2, vmax=2)
plt.title(f'Difference T2m (RFmodel - UrbClim): average plot \n{city} (2015-2017)\nBias {bias:.4f} {celcius}, RMSE: {rmse:.4f} {celcius}', fontsize=18)
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
plt.savefig(f'Average difference_plot_{city}_2015_2017.png', format='png')
plt.close()



#---------------------------------------------------------------------------------------------

cmap = 'seismic'

# Calculate the mean 'y_pred' value for each unique combination of 'x' and 'y'

print(mean_xarray_dataset)


# Print the resulting DataFrame with mean 'y_pred' values for each 'x' and 'y'
# Compute common min and max values
#vmin = min(mean_xarray_dataset['ypred_target'].min(), mean_xarray_dataset['ypred_target'].min())-0.7
#vmax = max(mean_xarray_dataset['ypred_target'].max(), mean_xarray_dataset['ypred_target'].max())+0.7

vmin=-2
vmax=2

# Plot 'y_pred' for the first timestep
plt.figure(figsize=(10, 8))
mappable_pred = mean_xarray_dataset['ypred_target'].plot(x='x', y='y', cmap=cmap, vmin=vmin, vmax=vmax, label='ypred_target')
plt.title(f'RFmodel predicted target variable \n average plot \n{city} (2015-2017)', fontsize=18)

plt.xlabel('Longitude', fontsize=18)
plt.ylabel('Latitude', fontsize=18)
# Adding direction indicators for x and y axes
x_ticks = np.linspace(plt.xticks()[0][0], plt.xticks()[0][-1], num=6)
y_ticks = np.linspace(plt.yticks()[0][0], plt.yticks()[0][-1], num=6)

x_direction = ['E' if tick > 0 else 'W' for tick in x_ticks]
y_direction = ['N' if tick > 0 else 'S' for tick in y_ticks]

x_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(x_ticks, x_direction)]
y_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(y_ticks, y_direction)]
# Set the modified tick labels with increased font size
plt.xticks(x_ticks, labels=x_tick_labels, fontsize=17)
plt.yticks(y_ticks, labels=y_tick_labels, fontsize=17)
mappable_pred.colorbar.ax.tick_params(labelsize=18)
mappable_pred.colorbar.set_label(f'Predicted target[{celcius}]', size=18)
plt.savefig(f'ypred_target_{city}_average_2015_2017.png', format='png')
plt.close()




# Plot 'y_test' for the first timestep with title 'UrbClim'
plt.figure(figsize=(10, 8))
mappable_test = mean_xarray_dataset['ytest_target'].plot(x='x', y='y', cmap=cmap, vmin=vmin, vmax=vmax, label='ytest_target')
plt.title(f'True target variable \n average plot \n{city} (2015-2017)' , fontsize=18)
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

mappable_test.colorbar.ax.tick_params(labelsize=18)
mappable_test.colorbar.set_label(f'test target[{celcius}]', size=18)
plt.savefig(f'ytest_target_{city}_average_2015_2017.png', format='png')
plt.close()


#-----------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np

# Define the colors for each land cover class

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np



import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

# Define the LCZ classes present in your dataset and their corresponding colors
lcz_colors = {
    1: '#8c0000',    # Compact high-rise
    2: '#d10000',    # Compact mid-rise
    3: '#ff0000',    # Compact low-rise
    4: '#bf4d00',    # Open high-rise
    5: '#ff6600',    # Open mid-rise
    6: '#ff9955',    # Open low-rise
    7: '#faee05',    # Lightweight low-rise
    8: '#bcbcbc',    # Large low-rise
    9: '#ffccaa',    # Sparsely built
    10: '#555555',   # Heavy industry
    11: '#006a00',   # Dense Trees (LCZ A)
    12: '#00aa00',   # Scattered Trees (LCZ B)
    13: '#648525',   # Bush, scrub (LCZ C)
    14: '#b9db79',   # Low plants (LCZ D)
    15: '#000000',   # Bare rock or paved (LCZ E)
    16: '#fbf7ae',   # Bare soil or sand (LCZ F)
    17: '#6a6aff'    # Water (LCZ G)
}

# Class names
lcz_labels = {
    1: 'Compact high-rise',
    2: 'Compact mid-rise',
    3: 'Compact low-rise',
    4: 'Open high-rise',
    5: 'Open mid-rise',
    6: 'Open low-rise',
    7: 'Lightweight low-rise',
    8: 'Large low-rise',
    9: 'Sparsely built',
    10: 'Heavy industry',
    11: 'Dense trees',
    12: 'Scattered trees',
    13: 'Bush, scrub',
    14: 'Low plants',
    15: 'Bare rock or paved',
    16: 'Bare soil or sand',
    17: 'Water'
}

mean_xarray_dataset['LCZ']=mean_xarray_dataset['LCZ']

# Extract the classes present in your dataset
present_classes = list(lcz_colors.keys())

# Create a custom colormap with only the present classes
lcz_cmap = mcolors.ListedColormap([lcz_colors[i] for i in present_classes])

# Plot LCZ for the city
plt.figure(figsize=(10, 8))
mappable_test = mean_xarray_dataset['LCZ'].plot(x='x', y='y', cmap=lcz_cmap, label='LCZ')
plt.title(f'LCZ plot for {city}', fontsize=18)
plt.xlabel('Longitude', fontsize=18)
plt.ylabel('Latitude', fontsize=18)

# Adjust ticks
x_ticks = np.linspace(plt.xticks()[0][0], plt.xticks()[0][-1], num=6)
y_ticks = np.linspace(plt.yticks()[0][0], plt.yticks()[0][-1], num=6)

x_direction = ['E' if tick > 0 else 'W' for tick in x_ticks]
y_direction = ['N' if tick > 0 else 'S' for tick in y_ticks]

# Define the degree symbol
x_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(x_ticks, x_direction)]
y_tick_labels = [f"{abs(round(tick, 1))}{degree}{dir}" for tick, dir in zip(y_ticks, y_direction)]

# Set the modified tick labels with increased font size
plt.xticks(x_ticks, labels=x_tick_labels, fontsize=17)
plt.yticks(y_ticks, labels=y_tick_labels, fontsize=17)

# Save the main plot
plt.savefig(f'LCZ_{city}.png', format='png')
plt.close()




# Create legend patches
legend_patches = [mpatches.Patch(color=lcz_colors[i], label=lcz_labels[i]) for i in present_classes]

# Create a separate figure for the legend
fig_legend = plt.figure(figsize=(3, 6))

# Add the legend to the figure
plt.legend(handles=legend_patches, loc='center', frameon=False, fontsize=12, handleheight=2)

# Remove axes for the legend figure
plt.gca().set_axis_off()
plt.gca().set_frame_on(False)

# Save the legend figure
plt.savefig(f'LCZ_legend_{city}.png', format='png')
plt.close()



# Create legend patches
legend_patches = [mpatches.Patch(color=lcz_colors[i], label=lcz_labels[i]) for i in present_classes]

# Create a separate figure for the legend
fig_legend = plt.figure(figsize=(15, 2))

# Add the legend to the figure
plt.legend(handles=legend_patches, loc='center', ncol=len(present_classes), frameon=False, fontsize=10, handleheight=2)

# Remove axes for the legend figure
plt.gca().set_axis_off()
plt.gca().set_frame_on(False)

# Save the legend figure
plt.savefig('LCZ_legend_horizontal.png', format='png')
plt.close()