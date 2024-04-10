import xarray as xr
import rasterio
import rioxarray as rxr
import os

file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/ERA_altitude/geopotential_ERA5.nc'
ds =xr.open_dataset(file_path)
ds['lon'] = (ds['lon'] + 180) % 360 - 180
ds.rio.write_crs('EPSG:4326', inplace=True)



# Function to replace "tas_" with "geopot" and remove the "_UrbClim_2015_09_v1.0" part
def modify_variable_string(input_string):
    modified_string = input_string.replace('tas_', 'geopot_')
    # Find the position of "_UrbClim_2015_09_v1.0" and remove it
    urbclim_index = modified_string.find('_UrbClim_2015_09_v1.0')
    if urbclim_index != -1:  # If found
        modified_string = modified_string[:urbclim_index]
        modified_string += '.nc'
    return modified_string


# Define the input directory and output directory
input_directory = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/Urbclim_data/UrbClim_reprojected'
output_directory = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/ERA_altitude'

# Make sure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Iterate through files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.nc'):  # Check if the file is a netCDF file
        filename_new= modify_variable_string(filename)

        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename_new)
  
        # Open the netCDF file
        city = xr.open_dataset(input_path)
        city = city.rio.write_crs('EPSG:4326')
        
        AHF_city=ds.interp_like(city, method='nearest')
        AHF_city.to_netcdf(output_path)





