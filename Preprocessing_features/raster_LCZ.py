import xarray as xr
import rasterio
import rioxarray as rxr
import os

file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/LCZ/lcz_filter_v2.tif'
ds = rxr.open_rasterio(file_path, parse_coordinates=True, masked=True)


# Function to replace "tas_" with "LCZ" and remove the "_UrbClim_2015_09_v1.0" part
def modify_variable_string(input_string):
    modified_string = input_string.replace('tas_', 'LCZ_')
    # Find the position of "_UrbClim_2015_09_v1.0" and remove it
    urbclim_index = modified_string.find('_UrbClim_2015_09_v1.0')
    if urbclim_index != -1:  # If found
        modified_string = modified_string[:urbclim_index]
        modified_string += '.nc'
    return modified_string


# Define the input directory and output directory
input_directory = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/Urbclim_data/UrbClim_reference'
output_directory = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/LCZ'

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

        LCZ_reproj = ds.rio.write_crs('EPSG:4326')
        min_lon=city['x'].min()-0.5
        min_lat=city['y'].min()-0.5
        max_lon=city['x'].max()+0.5
        max_lat=city['y'].max()+0.5

        LCZ_built=LCZ_reproj.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat)

        LCZ_built_city=LCZ_built.interp_like(city, method='nearest')
        LCZ_built_city.to_netcdf(output_path)





