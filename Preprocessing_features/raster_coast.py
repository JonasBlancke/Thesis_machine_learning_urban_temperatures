import xarray as xr
import rasterio
import rioxarray as rxr
import os



def extract_location(filename):
    # Find the start and end positions of the desired substring
    start_pos = filename.find("Coast_") + len("Coast_")
    end_pos = filename.find(".nc")

    # Check if both positions were found
    if start_pos != -1 and end_pos != -1:
        extracted_text = filename[start_pos:end_pos]
        return extracted_text
    else:
        return None








# Define the input directory and output directory
input_directory = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/coast/COAST'
output_directory = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/coast/COAST_reprojected'


# Make sure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Iterate through files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.nc'):  # Check if the file is a netCDF file
        input_path = os.path.join(input_directory, filename)
        output_path = os.path.join(output_directory, filename)
        location=extract_location(filename)
        input_2=f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/Urbclim_data/UrbClim_reprojected/tas_{location}_UrbClim_2015_09_v1.0.nc'
        city_dst=xr.open_dataset(input_2)

        # Open the netCDF file
        city = xr.open_dataset(input_path)
        city=city.interp_like(city_dst)

        # Save the reprojected dataset to the output path
        city.to_netcdf(output_path)
