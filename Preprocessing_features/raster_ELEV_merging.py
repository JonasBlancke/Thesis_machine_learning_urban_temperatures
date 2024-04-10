import xarray as xr
import rioxarray as rxr
import os

target_crs = 'EPSG:4326'

# Construct the paths for the corresponding TIFF files
first_part = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/elevation/DMET/50N000E_GMTED.tif'
second_part = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/elevation/DMET/50N030W_GMTED.tif'
third_part = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/elevation/DMET/30N030W_GMTED.tif'
fourth_part = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/elevation/DMET/30N000E_GMTED.tif'

# Open the corresponding TIFF files
first_data = rxr.open_rasterio(first_part, parse_coordinates=True).rio.write_crs(target_crs)
second_data = rxr.open_rasterio(second_part, parse_coordinates=True).rio.write_crs(target_crs)
third_data = rxr.open_rasterio(third_part, parse_coordinates=True).rio.write_crs(target_crs)
fourth_data = rxr.open_rasterio(fourth_part, parse_coordinates=True).rio.write_crs(target_crs)


# Combine datasets by coordinates with custom attribute merging
merged_data = xr.combine_by_coords(
    [first_data, second_data, third_data, fourth_data],
    combine_attrs='override')

# Save the merged data as a new GeoTIFF file
output_file = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/elevation/DMET/merged_DMET.tif'
merged_data.rio.to_raster(output_file)
print(f"Saved merged data to {output_file}")
