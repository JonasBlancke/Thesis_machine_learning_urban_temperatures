import xarray as xr
import rasterio
import rioxarray as rxr
import os

file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/elevation/DMET/merged_DMET.tif'
ds = rxr.open_rasterio(file_path, parse_coordinates=True, masked=True)


# Function to replace "tas_" with "LCZcorine" and remove the "_UrbClim_2015_09_v1.0" part
def modify_variable_string(input_string):
    modified_string = input_string.replace('tas_', 'DMETelev_')
    # Find the position of "_UrbClim_2015_09_v1.0" and remove it
    urbclim_index = modified_string.find('_UrbClim_2015_09_v1.0')
    if urbclim_index != -1:  # If found
        modified_string = modified_string[:urbclim_index]
        modified_string += '.nc'
    return modified_string


# Define the input directory and output directory
input_directory = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/Urbclim_data/UrbClim_reference'
output_directory = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/elevation/DMET/'

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
        crs_moll=ds.rio.crs
        city_mollweide=city.rio.reproject(crs_moll)
        
        min_lon=city_mollweide['x'].min()-500
        min_lat=city_mollweide['y'].min()-500
        max_lon=city_mollweide['x'].max()+500
        max_lat=city_mollweide['y'].max()+500
        subset = ds.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat)
        volume_reproj = subset.rio.reproject('EPSG:4326')

        min_lon=city['x'].min()
        min_lat=city['y'].min()
        max_lon=city['x'].max()
        max_lat=city['y'].max()

        volume_built=volume_reproj.rio.clip_box(minx=min_lon, miny=min_lat, maxx=max_lon, maxy=max_lat)

        volume_built_city=volume_built.interp_like(city)
        volume_built_city.to_netcdf(output_path)





