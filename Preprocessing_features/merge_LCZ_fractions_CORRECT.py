import xarray as xr
import numpy as np
import time
import pandas as pd

csv_file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_test/city_list.csv'
# Define the name of the column containing city names
city_column_name = 'City'
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
# Extract the city names from the specified column
cities = df[city_column_name].tolist()
#cities=['Bilbao', 'Antwerp']



for city in cities:
   # Load the NetCDF file
   start_time = time.perf_counter()
   nc_file_path = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/LCZ_high_res/WorldCover/NetCDF_Output/WorldCover_{city}.nc'
   ds = xr.open_dataset(nc_file_path)


   if 'band_data' in ds:
       # Extract values of band_data
       band_data_values = ds['band_data'].values

       # Create a new dataset with the desired structure
       ds_new = xr.Dataset(coords={'y': ds['y'], 'x': ds['x']})

       # Add band_data to the new dataset
       ds_new['land_cover'] = xr.DataArray(band_data_values, dims=['band', 'y', 'x'], coords={'y': ds['y'], 'x': ds['x']})


       # Copy attributes
       ds_new.attrs = ds.attrs

       # Print the resulting dataset
       ds=ds_new
       print(ds) 
  
   else:
       print("No 'band_data' variable present in the dataset.")







   with open('run_time2.txt', 'a') as f:
      f.write(f"ds: {ds} \n")
   # Get the first variable in the dataset
   variable_name = list(ds.variables.keys())[0]
   with open('run_time2.txt', 'a') as f:
      f.write(f"variable_name for {city}: {variable_name} \n")

   # Extract the data for the first variable
   #land_cover_data = ds[variable_name].values

   # Assuming 'land_cover' is the variable containing land cover information
   land_cover_data = ds['land_cover'].values

   # Load the "urbclim_berlin" dataset to get the original pixel size
   urbclim_file_path = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/Urbclim_data/UrbClim_reference/tas_{city}_UrbClim_2015_01_v1.0.nc'
   urbclim_ds = xr.open_dataset(urbclim_file_path)

   # Use x and y coordinates and get the original pixel size
   x_coords = urbclim_ds['x'].values
   y_coords = urbclim_ds['y'].values
   original_pixel_size_x = x_coords[1] - x_coords[0]
   original_pixel_size_y = y_coords[1] - y_coords[0]

   # Calculate the number of pixels in each dimension for the output
   output_shape_y = len(y_coords)
   output_shape_x = len(x_coords)

   # Initialize arrays to store fractions for each land cover type
   max_length = 0
   # Initialize 10 bands with zeros
   output_ds = xr.Dataset(coords={'y': y_coords, 'x': x_coords})

   for i in range(1, 11):
       output_ds[f'fraction_band_{i}'] = xr.DataArray(np.zeros((len(y_coords), len(x_coords))),
                                                       dims=['y', 'x'],
                                                       coords={'y': y_coords, 'x': x_coords})

   for i, y_value in enumerate(y_coords[:-1]):
       for j, x_value in enumerate(x_coords[:-1]):
           start_y = y_coords[i]
           end_y =  y_coords[i + 1]
           start_x = x_coords[j]
           end_x = x_coords[j + 1]
           clipped_ds = ds.sel(y=slice(start_y, end_y), x=slice(start_x, end_x))

           # Calculate unique values and their counts in the pixel
           unique, counts = np.unique(clipped_ds['land_cover'].values, return_counts=True)

           # Calculate fractions
           fractions = counts / np.sum(counts)

           # Map LCZ values to bands and assign fractions to output_ds
           for k, lcz_value in enumerate(range(10, 110, 10)):
               lcz_indices = np.where(unique == lcz_value)[0]
               band_index = k  # Initialize band_index outside the loop
   
               if len(lcz_indices) > 0:
                   band_name = f'fraction_band_{k+1}'
                   output_ds[band_name][i, j] = fractions[lcz_indices[0]]
               else:
                   band_name = f'fraction_band_{k+1}'
                   output_ds[band_name][i, j]= 0.0
   output_file_path = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/LCZ_high_res/fractions/fraction_{city}.nc'
   output_ds.to_netcdf(output_file_path)
   end_time = time.perf_counter()
   elapsed_time = end_time - start_time
   print(f"Elapsed time before loop: {elapsed_time} seconds")
   with open('run_time2.txt', 'a') as f:
       f.write(f"Elapsed time for {city} is: {elapsed_time} seconds\n")



