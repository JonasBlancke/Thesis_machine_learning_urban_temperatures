import xarray as xr
import rasterio
import rioxarray as rxr
import os


def extract_location(filename):
    # Find the start and end positions of the desired substring
    start_pos = filename.find("ruralurbanmask_") + len("ruralurbanmask_")
    end_pos = filename.find("_UrbClim_v1.0.nc")

    # Check if both positions were found
    if start_pos != -1 and end_pos != -1:
        extracted_text = filename[start_pos:end_pos]
        return extracted_text
    else:
        return None


input_directory = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/RURALORURBAN/urban_OR_rural'
output_directory = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/RURALORURBAN/URBANRURALreprojected'


# Make sure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Iterate through files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.nc'):  # Check if the file is a netCDF file
        location=extract_location(filename)
        input_2=f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/Urbclim_data/UrbClim_reprojected/tas_{location}_UrbClim_2015_09_v1.0.nc'
        city_dst=xr.open_dataset(input_2)
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        rural_org=xr.open_dataset(input_path)
        rural_org = rural_org.drop_vars(['longitude', 'latitude'])
        rural_org = rural_org.rio.write_crs('EPSG:3035')
        rural_reproj=rural_org.rio.reproject('EPSG:4326')
        rural_end=rural_reproj.interp_like(city_dst, method='nearest')
        rural_end.to_netcdf(output_path)





