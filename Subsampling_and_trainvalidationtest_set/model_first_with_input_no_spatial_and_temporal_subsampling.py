import xarray as xr
import rasterio
import rioxarray as rxr
import os
import numpy as np  
import pandas as pd
import math
import ephem
from timezonefinder import TimezoneFinder
import pytz
import pvlib
import datetime as datetime
import netCDF4 as nc
from glob import glob
import time
import random


#---------------------------
#STEP 1: LOAD ALL FILES:
#---------------------------

conversion_table = {
    1: [1],
    2: [2],
    3: [3, 4, 5, 6],
    4: [10, 11],
    5: [34],
    6: [8, 9, 30, 31, 38, 39],
    7: [7, 18, 26, 27, 32, 35, 36, 37],
    8: [12, 13, 14, 19, 20, 21],
    9: [15, 28, 29, 33],
    10: [16, 17, 22],
    11: [23, 25],
    12: [24],
    13: [40],
    14: [41],
    15: [42, 43, 44]
}








start_time = time.perf_counter()



def get_local_time(latitude, longitude, date_str):
    # Convert the date string to a datetime object in UTC
    date_obj_utc = pd.to_datetime(date_str).tz_localize('UTC')

    # Create an ephem Observer for a location using latitude and longitude
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.date = date_obj_utc

    # Calculate the timezone based on the location's coordinates
    tf = TimezoneFinder()
    local_tz_str = tf.timezone_at(lng=float(longitude), lat=float(latitude))
    local_timezone = pytz.timezone(local_tz_str)

    # Convert to the local time zone using pytz
    date_obj_local = observer.date.datetime().replace(tzinfo=pytz.utc).astimezone(local_timezone)
    local_hour = date_obj_local.hour

    return local_hour



def get_solar_position(latitude, longitude, date_str):
        # Convert the date string to a datetime object
        #date_obj = pd.to_datetime(date_str, format="%Y-%m-%d-%H").tz_localize('UTC')
        date_obj = pd.to_datetime(date_str).tz_localize('UTC')

        # Create an ephem Observer for a location using latitude and longitude
        observer = ephem.Observer()
        observer.lat = str(latitude)
        observer.lon = str(longitude)
        observer.date = date_obj
        sun = ephem.Sun()
        sun.compute(observer)

        # Get the declination angle in radians
        declination_radians = sun.dec

        # Convert the declination angle to degrees
        declination_degrees = ephem.degrees(declination_radians)

        # Calculate the timezone based on the location's coordinates
        tf = TimezoneFinder()
        #local_tz_str = tf.timezone_at(lng=float(longitude), lat=float(latitude))
        #local_timezone = pytz.timezone(local_tz_str)

        # Convert to the local time zone using pytz
        #date_obj_local = observer.date.datetime().astimezone(local_timezone)
        # Calculate solar position
        solar_position = pvlib.solarposition.get_solarposition(
            date_obj, latitude, longitude
        )
        elevation=solar_position['elevation'].iloc[0]
        return elevation, declination_degrees


def format_month(month):
    if month.startswith('0'):
        formatted_month=month[1]
        return formatted_month
    return month


def process_city_data(city, year, month):
    # Your code for loading, processing, and merging data, similar to the provided script, goes here
    # Final DataFrame after processing for the specific city, year, and month
    output_path = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_test/CITIES3/ready_for_model_{city}_{year}_{month}.csv'
    merged_df.to_csv(output_path, index=False)


csv_file_path = '/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_test/city_list.csv'
# Define the name of the column containing city names
city_column_name = 'City'
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)
# Extract the city names from the specified column



cities = df[city_column_name].tolist()
#years = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']  # Add your desired years here
#months = ['01', '02', '03', '04','05','06','07','08','09','10','11', '12']

cities = ['Gdansk']
years = [ '2015', '2016', '2017']
months = ['06','07','08']

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time before loop: {elapsed_time} seconds")
with open('run_time2.txt', 'a') as f:
    f.write(f"Elapsed time for begin step: {elapsed_time} seconds\n")


for city in cities:
    for year in years:
        for month in months:
            city_NDVI = city.split('_')[0]            
            start_time = time.perf_counter()

            formatted_month = format_month(month)
            print(formatted_month)
            print(month)
            #{city}:
            
            file_path = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/train_test/CITIES3/ready_for_model_{city}_{year}_{month}.csv'

            # Check if the file already exists
            if os.path.exists(file_path):
                print(f'{file_path} already present: doing nothing')
            else:
                # Run your script or perform desired operations
                print(f'Running script for {file_path}')
                # Your script goes here
                # ...
            #ERA

                ERA_dir = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/ERA/ERA5_{year}_{month}.nc'
                city_dir = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/Urbclim_data/UrbClim_reference/tas_{city}_UrbClim_2015_01_v1.0.nc'
                # Open the NetCDF file using netCDF4
                ERA5_pred = xr.open_dataset(ERA_dir)  # Predictors ERA5 September 2015


                file_path_urbclim = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/Urbclim_data/Urbclim_non_projected/tas_{city}_UrbClim_{year}_{month}_v1.0.nc'
                urbclim = xr.open_dataset(file_path_urbclim)
                 # Drop unnecessary variables
                urbclim = urbclim.drop_vars(['longitude', 'latitude'])
                # Set the CRS and reproject
                urbclim = urbclim.rio.write_crs('EPSG:3035')
                urbclim= urbclim.rio.reproject('EPSG:4326')

                dst = xr.open_dataset(city_dir)
                dst.rio.set_crs('EPSG:4326', inplace=True)

                # Convert time dimension to pandas datetime format
                urbclim['time'] = pd.to_datetime(urbclim['time'])
                ERA5_pred['time'] = pd.to_datetime(ERA5_pred['time'])


                # Identify overlapping time range
                overlap_start = max(urbclim.time.min(), ERA5_pred.time.min())
                overlap_end = min(urbclim.time.max(), ERA5_pred.time.max())

                print(overlap_start)
                print(overlap_end)

                # Convert to Pandas Timestamp objects
                overlap_start_pd = pd.Timestamp(overlap_start.values)
                overlap_end_pd = pd.Timestamp(overlap_end.values)

                # Calculate the total number of hours in the overlapping time range
                total_hours = (overlap_end_pd - overlap_start_pd).days * 24

                # Generate 24 unique random indices within the range of possible indices
                random_indices = random.sample(range(total_hours), 96)                              #96 hours randomly selected

                # Calculate the corresponding random times
                overlap_selected = [overlap_start_pd + pd.TimedeltaIndex([idx], unit='h') for idx in sorted(random_indices)]

                print("Randomly selected time range:")
                print(overlap_selected)

                # Convert overlap_selected to a list of Timestamps
                overlap_selected_timestamps = [ts[0] for ts in overlap_selected]









                # Filter datasets based on the selected time range
                #urbclim = urbclim.sel(time=overlap_selected_timestamps, method='nearest')
                #ERA5_pred_selected = ERA5_pred.sel(time=overlap_selected_timestamps, method='nearest')



                ERA5_pred_selected=ERA5_pred
                ERA = ERA5_pred_selected.interp(latitude=dst.y, longitude=dst.x, method='nearest')
                ERA = ERA.drop_vars(['longitude', 'latitude'])

                era_RH_dir=f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/ERA/ERA_RH/ERA5_RH_{year}_{month}.nc'
                ERA5_RH_pred = xr.open_dataset(era_RH_dir)  # Predictors ERA5 September 2015
                #ERA5_RH_pred=ERA5_RH_pred.sel(time=overlap_selected_timestamps, method='nearest')

                ERA5_RH_pred = ERA5_RH_pred.rio.write_crs("EPSG:4326")
                dst = xr.open_dataset(city_dir)
                dst.rio.set_crs('EPSG:4326', inplace=True)
                ERA_RH = ERA5_RH_pred.interp(latitude=dst.y, longitude=dst.x, method='nearest')
                ERA_RH = ERA_RH.drop_vars(['longitude', 'latitude'])


                era_LST_dir = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/ERA/LST/ERA5_LST_{year}_{month}.nc'
                ERA5_LST_pred = xr.open_dataset(era_LST_dir)  # Predictors ERA5 September 2015
                #ERA5_LST_pred=ERA5_LST_pred.sel(time=overlap_selected_timestamps, method='nearest')

                ERA5_LST_pred = ERA5_LST_pred.rio.write_crs("EPSG:4326")
                dst = xr.open_dataset(city_dir)
                dst.rio.set_crs('EPSG:4326', inplace=True)
                ERA_LST = ERA5_LST_pred.interp(latitude=dst.y, longitude=dst.x, method='nearest')
                ERA_LST = ERA_LST.drop_vars(['longitude', 'latitude'])


                era_SM_dir = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/ERA/soil_moisture/ERA5_soil_moisture_{year}_{month}.nc'
                ERA5_SM_pred = xr.open_dataset(era_SM_dir)  # Predictors ERA5 September 2015
                #ERA5_SM_pred=ERA5_SM_pred.sel(time=overlap_selected_timestamps, method='nearest')

                ERA5_SM_pred = ERA5_SM_pred.rio.write_crs("EPSG:4326")
                dst = xr.open_dataset(city_dir)
                dst.rio.set_crs('EPSG:4326', inplace=True)
                ERA_SM = ERA5_SM_pred.interp(latitude=dst.y, longitude=dst.x, method='nearest')
                ERA_SM = ERA_SM.drop_vars(['longitude', 'latitude'])


                era_CBH_dir = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/ERA/cloud_height/ERA5_cloud_base_height_{year}_{month}.nc'
                ERA5_CBH_pred = xr.open_dataset(era_CBH_dir)  # Predictors ERA5 September 2015
                #ERA5_CBH_pred=ERA5_CBH_pred.sel(time=overlap_selected_timestamps, method='nearest')

                ERA5_CBH_pred = ERA5_CBH_pred.rio.write_crs("EPSG:4326")
                dst = xr.open_dataset(city_dir)
                dst.rio.set_crs('EPSG:4326', inplace=True)
                ERA_CBH = ERA5_CBH_pred.interp(latitude=dst.y, longitude=dst.x, method='nearest')
                ERA_CBH = ERA_CBH.drop_vars(['longitude', 'latitude'])


                print("Time coordinates selected for urbclim:")
                print(urbclim['time'].values)

                print("Time coordinates selected for ERA5:")
                print(ERA['time'].values)
                #with open('run_time2.txt', 'a') as f:
                #    f.write(f"Time coordinates selected for urbclim:\n")
                #    f.write(f"{urbclim['time'].values}\n")
                #    f.write(f"Time coordinates selected for ERA5:\n")
                #    f.write(f"{ERA['time'].values}\n")
                #    f.write(f"Time coordinates selected for ERA5_CBH:\n")
                #    f.write(f"{ERA_CBH['time'].values}\n")


                file_path_LCZ_highres = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/LCZ_high_res/fractions/fraction_{city}.nc'
                LCZ_highres= xr.open_dataset(file_path_LCZ_highres)

                sum_1 = LCZ_highres['fraction_band_1']
                array1 = xr.DataArray(sum_1, coords={'y': LCZ_highres['y'], 'x': LCZ_highres['x']}, dims=['y', 'x'])
                LCZ_tree=  xr.Dataset({'LCZtree': array1})

                sum_5 = LCZ_highres['fraction_band_5']
                array5 = xr.DataArray(sum_5, coords={'y': LCZ_highres['y'], 'x': LCZ_highres['x']}, dims=['y', 'x'])
                LCZ_built= xr.Dataset({'LCZbuilt': array5})

                sum_8 = LCZ_highres['fraction_band_8']
                array8 = xr.DataArray(sum_8, coords={'y': LCZ_highres['y'], 'x': LCZ_highres['x']}, dims=['y', 'x'])
                LCZ_water= xr.Dataset({'LCZwater': array8})

                sum_2346910 = LCZ_highres['fraction_band_2']+LCZ_highres['fraction_band_3']+LCZ_highres['fraction_band_4']+LCZ_highres['fraction_band_6']+LCZ_highres['fraction_band_9']+LCZ_highres['fraction_band_10']
                array2346910 = xr.DataArray(sum_2346910, coords={'y': LCZ_highres['y'], 'x': LCZ_highres['x']}, dims=['y', 'x'])
                LCZ_bareandveg=  xr.Dataset({'LCZbareandveg': array2346910})

                #LCZ_bareandveg
                #LCZ_tree
                #LCZ_water
                #LCZ_built
      


                file_path_LC_CORINE = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/CORINE/LCZcorine_{city}_UrbClim_2015_01_v1.0.nc'
                LC_CORINE= xr.open_dataset(file_path_LC_CORINE)
                # Reclassify the variable using the conversion table
                reclassified_variable = LC_CORINE['__xarray_dataarray_variable__'].copy()

                for new_value, old_values in conversion_table.items():
                   reclassified_variable = reclassified_variable.where(~reclassified_variable.isin(old_values), new_value)

                # Print structure of reclassified variable
                print(reclassified_variable)

                # Update the original dataset with the reclassified variable
                LC_CORINE['__xarray_dataarray_variable__'] = reclassified_variable
                LC_CORINE = LC_CORINE.rename({'__xarray_dataarray_variable__': 'LC_CORINE'})






                file_path_geopot = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/ERA_altitude/geopot_{city}_UrbClim_2015_01_v1.0.nc'
                geopot = xr.open_dataset(file_path_geopot)

                file_path_AHF = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/AHF/AHF_{city}.nc'
                AHF = xr.open_dataset(file_path_AHF)
                file_path_albedo = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/albedo/albedo_{city}_UrbClim_2015_01_v1.0.nc'
                albedo = xr.open_dataset(file_path_albedo)
                file_path_height = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/height/height_{city}.nc'
                height= xr.open_dataset(file_path_height)
                file_path_surface = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/surface/surface_{city}.nc'
                surface = xr.open_dataset(file_path_surface)
                file_path_coast = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/coast/COAST_reprojected/Coast_{city}.nc'
                coast = xr.open_dataset(file_path_coast)
                #file_path_elevation = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/elevation/mood/elevation_{city}_UrbClim_2015_01_v1.0.nc'
                #elevation = xr.open_dataset(file_path_elevation)
                file_path_elevation = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/elevation/DMET/DMETelev_{city}_UrbClim_2015_01_v1.0.nc'
                elevation = xr.open_dataset(file_path_elevation)


                file_path_imperv = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/imperv/imperv_{city}_UrbClim_2015_01_v1.0.nc'
                imperv = xr.open_dataset(file_path_imperv)
                file_path_landseamask = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/LANDWATER/LANDSEA_reprojected/landseamask_{city}_UrbClim_v1.0.nc'
                landseamask = xr.open_dataset(file_path_landseamask)   
                file_path_ruralurbanmask = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/RURALORURBAN/URBANRURALreprojected/ruralurbanmask_{city}_UrbClim_v1.0.nc'     
                ruralurbanmask = xr.open_dataset(file_path_ruralurbanmask)
                file_path_LCZ = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/LCZ/LCZ_{city}_UrbClim_2015_01_v1.0.nc'
                LCZ = xr.open_dataset(file_path_LCZ)
                file_path_population = f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/population/population_{city}.nc'
                population = xr.open_dataset(file_path_population)
                file_path_NDVI=f'/kyukon/data/gent/vo/000/gvo00041/vsc46127/NDVI/GGE/NDVI_cities/{year}_{formatted_month}/{city_NDVI}_NDVI_{year}_{formatted_month}'
                NDVI=xr.open_dataset(file_path_NDVI)


                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Elapsed time for step 1: {elapsed_time} seconds")
                with open('run_time2.txt', 'a') as f:
                    f.write(f"Elapsed time for step 1: {elapsed_time} seconds\n")


                #---------------------------
                #STEP 2: CONVERT TO DATAFRAME
                #---------------------------


                #CHANGE THE NAMES!!!!



                start_time = time.perf_counter()

                #{city}--------------------------------------------------------------------------------------
                print("Urbclim size:", urbclim.sizes)
                mask = urbclim['tas']
                mask_urbclim = mask.isel(time=0)
                print(mask_urbclim.notnull())
                # Assuming your DataArray is named mask_urbclim
                #mask_urbclim = mask_urbclim.drop(['time', 'spatial_ref'])
                print(mask_urbclim.notnull())

                mask_NDVI=NDVI['__xarray_dataarray_variable__']
                mask_NDVI =mask_NDVI.squeeze("band")
                mask_NDVI=mask_NDVI.drop_vars(['band'])
                print(mask_NDVI)
                print("mask_NDVI size:", mask_NDVI.sizes)
                rural_urban_mask=ruralurbanmask['ruralurbanmask']
                land_water_mask=landseamask['landseamask']

                print(urbclim)
                masked_data = urbclim.where(mask_urbclim.notnull(), drop=True)
                #urbclim_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #urbclim_masked=urbclim_masked.where(mask_NDVI.notnull(), drop=True)
                #df_urbclim=urbclim_masked.to_dataframe()
                df_urbclim=masked_data.to_dataframe()
                print(df_urbclim)




                masked_data = geopot.where(mask_urbclim.notnull(), drop=True)
                #urbclim_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #urbclim_masked=urbclim_masked.where(mask_NDVI.notnull(), drop=True)
                #df_urbclim=urbclim_masked.to_dataframe()
                df_geopot=masked_data.to_dataframe()
                df_geopot.rename(columns={'z': 'GEOPOT'}, inplace=True)
                print('this is geopot df')
                print(df_geopot)



                print("Population shape:", population.sizes)
                print("Mask shape:", mask_urbclim.sizes)

                masked_data= population.where(mask_urbclim.notnull(), drop=True)
                #population_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #population_masked=population_masked.where(mask_NDVI.notnull(), drop=True)
                #df_population=population_masked.to_dataframe()
                df_population=masked_data.to_dataframe()
                print(df_population)

                df_population.rename(columns={'__xarray_dataarray_variable__': 'POP'}, inplace=True)

                print(AHF) 
                print(mask_urbclim)
                print(mask)

                # Assuming mask_urbclim is a DataArray and threshold is the threshold value
                #mask = (mask_urbclim > threshold) & ~mask_urbclim.isnull()

                # Use the where method to filter the AHF dataset
                #masked_data = AHF.where(mask, drop=True)
                # Apply the mask to the AHF dataset using boolean indexing
                #masked_data = AHF[mask]

                masked_data=AHF.where(mask_urbclim.notnull(), drop=True)
                #AHF_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #AHF_masked=AHF_masked.where(mask_NDVI.notnull(), drop=True)
                #df_AHF=AHF_masked.to_dataframe()
                df_AHF=masked_data.to_dataframe()
                print(df_AHF)

                df_AHF.rename(columns={'__xarray_dataarray_variable__': 'AHF'}, inplace=True)

                masked_data = LCZ.where(mask_urbclim.notnull(), drop=True)
                #LCZ_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #LCZ_masked=LCZ_masked.where(mask_NDVI.notnull(), drop=True)
                #df_LCZ=LCZ_masked.to_dataframe()
                df_LCZ=masked_data.to_dataframe()
                df_LCZ.rename(columns={'__xarray_dataarray_variable__': 'LCZ'}, inplace=True)




                masked_data = ruralurbanmask.where(mask_urbclim.notnull(), drop=True)
                #LCZ_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #LCZ_masked=LCZ_masked.where(mask_NDVI.notnull(), drop=True)
                #df_LCZ=LCZ_masked.to_dataframe()
                df_ruralurban=masked_data.to_dataframe()

                masked_data = landseamask.where(mask_urbclim.notnull(), drop=True)
                #LCZ_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #LCZ_masked=LCZ_masked.where(mask_NDVI.notnull(), drop=True)
                #df_LCZ=LCZ_masked.to_dataframe()
                df_landsea=masked_data.to_dataframe()



                masked_data = coast.where(mask_urbclim.notnull(), drop=True)
                #coast_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #coast_masked=coast_masked.where(mask_NDVI.notnull(), drop=True)
                #df_coast=coast_masked.to_dataframe()
                df_coast=masked_data.to_dataframe()
                df_coast.rename(columns={'__xarray_dataarray_variable__': 'COAST'}, inplace=True)




                #LC_CORINE
                masked_data = LC_CORINE.where(mask_urbclim.notnull(), drop=True)
                #LC_CORINE_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #LC_CORINE_masked=LC_CORINE_masked.where(mask_NDVI.notnull(), drop=True)
                #LC_CORINE=LC_CORINE_masked.to_dataframe()
                df_LC_CORINE=masked_data.to_dataframe()








                #LCZ_bareandveg
                masked_data = LCZ_bareandveg.where(mask_urbclim.notnull(), drop=True)
                #LCZ_bareandveg_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #LCZ_bareandveg_masked=LCZ_bareandveg_masked.where(mask_NDVI.notnull(), drop=True)
                #df_LCZ_bareandveg=LCZ_bareandveg_masked.to_dataframe()
                df_LCZ_bareandveg=masked_data.to_dataframe()
                df_LCZ_bareandveg.rename(columns={'LCZbareandveg': 'LC_BAREANDVEG'}, inplace=True)

                #LCZ_tree
                masked_data = LCZ_tree.where(mask_urbclim.notnull(), drop=True)
                #LCZ_tree_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #LCZ_tree_masked=LCZ_tree_masked.where(mask_NDVI.notnull(), drop=True)
                #df_LCZ_tree=LCZ_tree_masked.to_dataframe()
                df_LCZ_tree=masked_data.to_dataframe()
                df_LCZ_tree.rename(columns={'LCZtree': 'LC_TREE'}, inplace=True)

                #LCZ_water
                masked_data = LCZ_water.where(mask_urbclim.notnull(), drop=True)
                #LCZ_water_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #LCZ_water_masked=LCZ_water_masked.where(mask_NDVI.notnull(), drop=True)
                #df_LCZ_water=LCZ_water_masked.to_dataframe()
                df_LCZ_water=masked_data.to_dataframe()
                df_LCZ_water.rename(columns={'LCZwater': 'LC_WATER'}, inplace=True)

                #LCZ_built
                masked_data = LCZ_built.where(mask_urbclim.notnull(), drop=True)
                #LCZ_built_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #LCZ_built_masked=LCZ_built_masked.where(mask_NDVI.notnull(), drop=True)
                #df_LCZ_built=LCZ_built_masked.to_dataframe()
                df_LCZ_built=masked_data.to_dataframe()
                df_LCZ_built.rename(columns={'LCZbuilt': 'LC_BUILT'}, inplace=True)



                masked_data = height.where(mask_urbclim.notnull(), drop=True)
                #height_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #height_masked=height_masked.where(mask_NDVI.notnull(), drop=True)
                #df_height=height_masked.to_dataframe()
                df_height=masked_data.to_dataframe()
                df_height.rename(columns={'__xarray_dataarray_variable__': 'HEIGHT'}, inplace=True)

                masked_data = imperv.where(mask_urbclim.notnull(), drop=True)
                #imperv_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #imperv_masked=height_masked.where(mask_NDVI.notnull(), drop=True)
                #df_imperv=height_masked.to_dataframe()
                df_imperv=masked_data.to_dataframe()
                df_imperv.rename(columns={'__xarray_dataarray_variable__': 'IMPERV'}, inplace=True)

                masked_data = elevation.where(mask_urbclim.notnull(), drop=True)
                #elevation_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #elevation_masked=height_masked.where(mask_NDVI.notnull(), drop=True)
                #df_elevation=height_masked.to_dataframe()
                df_elevation=masked_data.to_dataframe()
                df_elevation.rename(columns={'__xarray_dataarray_variable__': 'ELEV'}, inplace=True)


                masked_data = surface.where(mask_urbclim.notnull(), drop=True)
                #surface_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #surface_masked=surface_masked.where(mask_NDVI.notnull(), drop=True)
                #df_surface=surface_masked.to_dataframe()
                df_surface=masked_data.to_dataframe()
                df_surface.rename(columns={'__xarray_dataarray_variable__': 'BUILT_FRAC'}, inplace=True)

                NDVI=NDVI['__xarray_dataarray_variable__']
                NDVI =NDVI.squeeze("band")
                NDVI=NDVI.drop_vars(['band'])
                masked_data = NDVI.where(mask_urbclim.notnull(), drop=True)
                #NDVI_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #NDVI_masked=NDVI_masked.where(mask_NDVI.notnull(), drop=True)
                #df_NDVI=NDVI_masked.to_dataframe()
                df_NDVI=masked_data.to_dataframe()
                df_NDVI.rename(columns={'__xarray_dataarray_variable__': 'NDVI'}, inplace=True)


                new_datasets = {}
                # Iterate over the variables in the original Dataset
                for var_name in ERA.data_vars:
                    var_data = ERA[var_name]
                    new_dataset = xr.Dataset({var_name: var_data})
                    new_datasets[var_name] = new_dataset
                # Now, you have a dictionary of new xarray Datasets, each with a single variable
                # You can access each new Dataset like this:
                df_t2m = new_datasets['t2m']
                df_blh = new_datasets['blh']
                df_sp = new_datasets['sp']
                df_tp = new_datasets['tp']
                df_tcc=new_datasets['tcc']
                df_ssr=new_datasets['ssr']
                df_wind_u=new_datasets['u10']
                df_wind_v=new_datasets['v10']
                df_cape=new_datasets['cape']

                new_datasets = {}
                # Iterate over the variables in the original Dataset
                for var_name in ERA_RH.data_vars:
                    print("Variable Name:", var_name)
                    var_data = ERA_RH[var_name]
                    new_dataset = xr.Dataset({var_name: var_data})
                    new_datasets[var_name] = new_dataset
                # Now, you have a dictionary of new xarray Datasets, each with a single variable
                # You can access each new Dataset like this:
                df_RH = new_datasets['r']



                new_datasets = {}
                # Iterate over the variables in the original Dataset
                for var_name in ERA_LST.data_vars:
                    print("Variable Name:", var_name)
                    var_data = ERA_LST[var_name]
                    new_dataset = xr.Dataset({var_name: var_data})
                    new_datasets[var_name] = new_dataset
                # Now, you have a dictionary of new xarray Datasets, each with a single variable
                # You can access each new Dataset like this:
                df_LST = new_datasets['skt']


                new_datasets = {}
                # Iterate over the variables in the original Dataset
                for var_name in ERA_CBH.data_vars:
                    print("Variable Name:", var_name)
                    var_data = ERA_CBH[var_name]
                    new_dataset = xr.Dataset({var_name: var_data})
                    new_datasets[var_name] = new_dataset
                # Now, you have a dictionary of new xarray Datasets, each with a single variable
                # You can access each new Dataset like this:
                df_CBH = new_datasets['cbh']


                new_datasets = {}
                # Iterate over the variables in the original Dataset
                for var_name in ERA_SM.data_vars:
                    print("Variable Name:", var_name)
                    var_data = ERA_SM[var_name]
                    new_dataset = xr.Dataset({var_name: var_data})
                    new_datasets[var_name] = new_dataset
                # Now, you have a dictionary of new xarray Datasets, each with a single variable
                # You can access each new Dataset like this:
                df_SM = new_datasets['swvl1']







                masked_data = df_t2m.where(mask_urbclim.notnull(), drop=True)
                #t2m_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #t2m_masked=t2m_masked.where(mask_NDVI.notnull(), drop=True)

                df_t2m=masked_data.to_dataframe()


                masked_data = df_ssr.where(mask_urbclim.notnull(), drop=True)
                #ssr_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #ssr_masked=ssr_masked.where(mask_NDVI.notnull(), drop=True)
                df_ssr=masked_data.to_dataframe()
    
                masked_data = df_tcc.where(mask_urbclim.notnull(), drop=True)
                #tcc_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #tcc_masked=tcc_masked.where(mask_NDVI.notnull(), drop=True)
                df_tcc=masked_data.to_dataframe()

                masked_data = df_cape.where(mask_urbclim.notnull(), drop=True)
                #cape_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #cape_masked=cape_masked.where(mask_NDVI.notnull(), drop=True)
                df_cape=masked_data.to_dataframe()



                masked_data = df_blh.where(mask_urbclim.notnull(), drop=True)
                #blh_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #blh_masked=blh_masked.where(mask_NDVI.notnull(), drop=True)
                df_blh=masked_data.to_dataframe()

                masked_data = df_sp.where(mask_urbclim.notnull(), drop=True)
                #sp_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #sp_masked=sp_masked.where(mask_NDVI.notnull(), drop=True)
                df_sp=masked_data.to_dataframe()

                masked_data = df_tp.where(mask_urbclim.notnull(), drop=True)
                #tp_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #tp_masked=tp_masked.where(mask_NDVI.notnull(), drop=True)
                df_tp=masked_data.to_dataframe()


                masked_data = df_wind_u.where(mask_urbclim.notnull(), drop=True)
                #wind_u_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #wind_u_masked=wind_u_masked.where(mask_NDVI.notnull(), drop=True)
                df_wind_u=masked_data.to_dataframe()

                masked_data = df_wind_v.where(mask_urbclim.notnull(), drop=True)
                #wind_v_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #wind_v_masked=wind_v_masked.where(mask_NDVI.notnull(), drop=True)
                df_wind_v=masked_data.to_dataframe()

                masked_data = df_RH.where(mask_urbclim.notnull(), drop=True)
                #RH_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #RH_masked=RH_masked.where(mask_NDVI.notnull(), drop=True)
                df_RH=masked_data.to_dataframe()

                masked_data = df_LST.where(mask_urbclim.notnull(), drop=True)
                #RH_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #RH_masked=RH_masked.where(mask_NDVI.notnull(), drop=True)
                df_LST=masked_data.to_dataframe()

                masked_data = df_SM.where(mask_urbclim.notnull(), drop=True)
                #RH_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #RH_masked=RH_masked.where(mask_NDVI.notnull(), drop=True)
                df_SM=masked_data.to_dataframe()

                masked_data = df_CBH.where(mask_urbclim.notnull(), drop=True)
                #RH_masked=masked_data.where(land_water_mask.notnull(), drop=True)
                #RH_masked=RH_masked.where(mask_NDVI.notnull(), drop=True)
                df_CBH=masked_data.to_dataframe()


                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Elapsed time for step 2: {elapsed_time} seconds")
                with open('run_time2.txt', 'a') as f:
                    f.write(f"Elapsed time for step 2: {elapsed_time} seconds\n")



                #---------------------------------------------------------------------------------------------
                #STEP 3: MERGING
                #---------------------------------------------------------------------------------------------
                #MAKE SURE UNITS ARE OK

                start_time = time.perf_counter()

                #{city}


                df_NDVI['NDVI'] /= 10000
                df_NDVI=df_NDVI.drop(['spatial_ref'], axis=1)
                df_NDVI.reset_index(inplace=True)
                df_NDVI=df_NDVI[['NDVI', 'y', 'x']]
                print(df_NDVI)

                df_sp=df_sp.drop(['spatial_ref'], axis=1)
                df_sp.reset_index(inplace=True)
                df_sp=df_sp[['sp', 'y', 'x', 'time']].rename(columns={'sp': 'SP'})
                print(df_sp)

                df_tp['tp'] *= 10000
                df_tp=df_tp.drop(['spatial_ref'], axis=1)
                df_tp.reset_index(inplace=True)
                df_tp=df_tp[['tp', 'y', 'x', 'time']].rename(columns={'tp': 'PRECIP'})
                print(df_tp)

                df_blh=df_blh.drop(['spatial_ref'], axis=1)
                df_blh.reset_index(inplace=True)
                df_blh=df_blh[['blh', 'y', 'x', 'time']].rename(columns={'blh': 'BLH'})

                df_wind_u=df_wind_u.drop(['spatial_ref'], axis=1)
                df_wind_u.reset_index(inplace=True)
                df_wind_u=df_wind_u[['u10', 'y', 'x', 'time']].rename(columns={'u10': 'U10'})

                df_wind_v=df_wind_v.drop(['spatial_ref'], axis=1)
                df_wind_v.reset_index(inplace=True)
                df_wind_v=df_wind_v[['v10', 'y', 'x', 'time']].rename(columns={'v10': 'V10'})
            
                df_WS=df_wind_u
                df_WS['wind_speed'] = np.sqrt(df_wind_u['U10']**2 + df_wind_v['V10']**2)
                # Drop unnecessary columns and reset index
                df_WS = df_WS[['wind_speed', 'y', 'x', 'time']].rename(columns={'wind_speed': 'WS'})
                df_WS.reset_index(drop=True, inplace=True)









                df_RH=df_RH.drop(['spatial_ref'], axis=1)
                df_RH['RH']=df_RH['r']
                df_RH.reset_index(inplace=True)
                df_RH=df_RH[['RH', 'y', 'x', 'time']]


                df_LST=df_LST.drop(['spatial_ref'], axis=1)
                df_LST['T_SK']=df_LST['skt']
                df_LST.reset_index(inplace=True)
                df_LST=df_LST[['T_SK', 'y', 'x', 'time']]

                df_CBH=df_CBH.drop(['spatial_ref'], axis=1)
                df_CBH['CBH']=df_CBH['cbh']
                df_CBH.reset_index(inplace=True)
                df_CBH=df_CBH[['CBH', 'y', 'x', 'time']]

                df_SM=df_SM.drop(['spatial_ref'], axis=1)
                df_SM['SM']=df_SM['swvl1']
                df_SM.reset_index(inplace=True)
                df_SM=df_SM[['SM', 'y', 'x', 'time']]



                df_t2m=df_t2m.drop(['spatial_ref'], axis=1)
                df_t2m.reset_index(inplace=True)
                df_t2m=df_t2m[['t2m', 'y', 'x', 'time']].rename(columns={'t2m': 'T_2M'})

                df_tcc=df_tcc.drop(['spatial_ref'], axis=1)
                df_tcc.reset_index(inplace=True)
                df_tcc=df_tcc[['tcc', 'y', 'x', 'time']].rename(columns={'tcc': 'TCC'})

                df_cape=df_cape.drop(['spatial_ref'], axis=1)
                df_cape.reset_index(inplace=True)
                df_cape=df_cape[['cape', 'y', 'x', 'time']].rename(columns={'cape': 'CAPE'})

                df_ssr=df_ssr.drop(['spatial_ref'], axis=1)
                df_ssr.reset_index(inplace=True)
                df_ssr=df_ssr[['ssr', 'y', 'x', 'time']].rename(columns={'ssr': 'SSR'})

                df_surface['BUILT_FRAC'] /= 10000
                df_surface=df_surface.drop(['spatial_ref', 'time'], axis=1)
                df_surface.reset_index(inplace=True)
                df_surface=df_surface[['BUILT_FRAC', 'y', 'x']]
                print(df_surface)



                df_height=df_height.drop(['spatial_ref', 'time'], axis=1)
                df_height.reset_index(inplace=True)
                df_height=df_height[['HEIGHT', 'y', 'x']]

                df_LC_CORINE=df_LC_CORINE.drop(['spatial_ref', 'time'], axis=1)
                df_LC_CORINE.reset_index(inplace=True)
                df_LC_CORINE=df_LC_CORINE[['LC_CORINE', 'y', 'x']]




                #LCZ_bareandveg
                df_LCZ_bareandveg=df_LCZ_bareandveg.drop(['spatial_ref', 'time'], axis=1)
                df_LCZ_bareandveg.reset_index(inplace=True)
                df_LCZ_bareandveg=df_LCZ_bareandveg[['LC_BAREANDVEG', 'y', 'x']]

                #LCZ_tree
                df_LCZ_tree=df_LCZ_tree.drop(['spatial_ref', 'time'], axis=1)
                df_LCZ_tree.reset_index(inplace=True)
                df_LCZ_tree=df_LCZ_tree[['LC_TREE', 'y', 'x']]

                #LCZ_water
                df_LCZ_water=df_LCZ_water.drop(['spatial_ref', 'time'], axis=1)
                df_LCZ_water.reset_index(inplace=True)
                df_LCZ_water=df_LCZ_water[['LC_WATER', 'y', 'x']]

                #LCZ_built
                df_LCZ_built=df_LCZ_built.drop(['spatial_ref', 'time'], axis=1)
                df_LCZ_built.reset_index(inplace=True)
                df_LCZ_built=df_LCZ_built[['LC_BUILT', 'y', 'x']]

                df_elevation=df_elevation.drop(['spatial_ref', 'time'], axis=1)
                df_elevation.reset_index(inplace=True)
                
                df_elevation=df_elevation[['ELEV', 'y', 'x']]


                df_coast['COAST'] /= 1000
                df_coast=df_coast.drop(['spatial_ref', 'time'], axis=1)
                df_coast.reset_index(inplace=True)
                df_coast=df_coast[['COAST', 'y', 'x']]
                df_coast=df_coast.dropna()                          #EXCLUDE SEA PIXELS FROM ANALYSIS


                df_imperv=df_imperv.drop(['spatial_ref', 'time'], axis=1)
                df_imperv.reset_index(inplace=True)
                df_imperv=df_imperv[['IMPERV', 'y', 'x']]

                df_AHF=df_AHF.drop(['spatial_ref', 'time'], axis=1)
                df_AHF.reset_index(inplace=True)
                df_AHF=df_AHF[['AHF', 'y', 'x']]
                print(df_AHF)

                df_population=df_population.drop(['spatial_ref', 'time'], axis=1)
                df_population.reset_index(inplace=True)
                df_population=df_population[['POP', 'y', 'x']]

                df_LCZ=df_LCZ.drop(['spatial_ref', 'time'], axis=1)
                df_LCZ.reset_index(inplace=True)
                df_LCZ=df_LCZ[['LCZ', 'y', 'x']]

                df_landsea=df_landsea.drop(['spatial_ref'], axis=1)
                df_landsea.reset_index(inplace=True)
                df_landsea=df_landsea[['landseamask', 'y', 'x']]
                df_landsea.fillna(0, inplace=True)

                df_ruralurban=df_ruralurban.drop(['spatial_ref'], axis=1)
                df_ruralurban.reset_index(inplace=True)
                df_ruralurban=df_ruralurban[['ruralurbanmask', 'y', 'x']]
                df_ruralurban.fillna(0, inplace=True)


                df_geopot=df_geopot.drop(['spatial_ref'], axis=1)
                df_geopot.reset_index(inplace=True)
                df_geopot=df_geopot[['GEOPOT', 'y', 'x']]




                df_urbclim=df_urbclim.drop(['spatial_ref'], axis=1)
                df_urbclim.reset_index(inplace=True)
                df_urbclim=df_urbclim[['tas', 'y', 'x', 'time']].rename(columns={'tas': 'T_TARGET'})



                # Merge the first two dataframes based on 'x', 'y', and time
                merged_df = pd.merge(df_sp, df_blh, on=['x', 'y', 'time'], how='inner')
                print("Size of first merge:", merged_df.shape)

                merged_df = pd.merge(merged_df, df_tp, on=['x', 'y', 'time'], how='inner')
                print("Size of before urbclim:", merged_df.shape)

                merged_df = pd.merge(merged_df, df_urbclim, on=['x', 'y', 'time'], how='inner')
                merged_df = pd.merge(merged_df, df_t2m, on=['x', 'y', 'time'], how='inner')
                merged_df = pd.merge(merged_df, df_ssr, on=['x', 'y', 'time'], how='inner')
                merged_df = pd.merge(merged_df, df_tcc, on=['x', 'y', 'time'], how='inner')
                merged_df = pd.merge(merged_df, df_cape, on=['x', 'y', 'time'], how='inner')
                merged_df = pd.merge(merged_df, df_RH, on=['x', 'y', 'time'], how='inner')
                merged_df = pd.merge(merged_df, df_wind_u, on=['x', 'y', 'time'], how='inner')
                merged_df = pd.merge(merged_df, df_wind_v, on=['x', 'y', 'time'], how='inner')
                merged_df = pd.merge(merged_df, df_LST, on=['x', 'y', 'time'], how='inner')
                merged_df = pd.merge(merged_df, df_SM, on=['x', 'y', 'time'], how='inner')
                merged_df = pd.merge(merged_df, df_CBH, on=['x', 'y', 'time'], how='inner')
                merged_df = pd.merge(merged_df, df_WS, on=['x', 'y', 'time'], how='inner')  

                print("Size of merged_time:", merged_df.shape)

            # Continue merging with other dataframes
                merged_df = pd.merge(merged_df, df_surface, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_population, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_NDVI, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_height, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_coast, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_LCZ_tree, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_LCZ_water, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_LCZ_built, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_LCZ_bareandveg, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_LC_CORINE, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_AHF, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_LCZ, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_imperv, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_elevation, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_geopot, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_ruralurban, on=['x', 'y'], how='inner')
                merged_df = pd.merge(merged_df, df_landsea, on=['x', 'y'], how='inner')


                merged_df = merged_df[merged_df['T_TARGET'] < 350]
                
                merged_df.reset_index(inplace=True)

                numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
                merged_df[numeric_columns] = merged_df[numeric_columns].astype(np.float32)


                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Elapsed time for step 3: {elapsed_time} seconds")
                with open('run_time2.txt', 'a') as f:
                    f.write(f"Elapsed time for step 3: {elapsed_time} seconds\n")

                #-------------------------
                #STEP 4: adding local time, solar elevation and declination
                #-------------------------




                #{city}


                merged_df=merged_df
                merged_df['time'] = pd.to_datetime(merged_df['time'], format="%Y-%m-%d %H:%M:%S")
                #merged_df['time'] = merged_df['time'].strftime("%Y-%m-%d")
                merged_df['hour'] = merged_df['time'].dt.hour
                merged_df['day'] = merged_df['time'].dt.day
                merged_df['month'] = merged_df['time'].dt.month
                merged_df['year'] = merged_df['time'].dt.year




                start_time = time.perf_counter()
                data=merged_df

                data['T_2M_SL']=data['T_2M']+6.5*((data['GEOPOT']/ 9.80665))/1000

                
               




                #MODIFY THIS
                columns_to_mean = ['SP', 'BLH', 'SSR', 'TCC', 'U10', 'V10','WS', 'PRECIP', 'CAPE', 'T_2M', 'RH', 'SM', 'T_SK', 'CBH', 'T_2M_SL']     

    
                data = data[data['T_TARGET'] <= 350]
                data['city']=city
                numerical_cols = data.select_dtypes(include=['float64']).columns
                data[numerical_cols] = data[numerical_cols].astype('float32')
                # Calculate means for specified columns

                data[columns_to_mean] = data.groupby('time')[columns_to_mean].transform('mean')
           
                #grouped_data = data.groupby(['LCZ', 'hour'])

                # Step 2: Select 2% of the rows from each group
                #selected_rows = grouped_data.apply(lambda x: x.sample(frac=0.02))
    
                # Resetting the index before merging
                #selected_rows.reset_index(drop=True, inplace=True)
                #mergecols=data[['y', 'x', 'time', 'LCZ', 'hour']]
                # Step 3: Merge the selected rows together
                #merged_df = selected_rows.merge(mergecols, how='inner', on=['y', 'x', 'time', 'LCZ', 'hour'])
                # Display the original DataFrame



                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Elapsed time for step 4: {elapsed_time} seconds")
                with open('run_time2.txt', 'a') as f:
                    f.write(f"Elapsed time for step 4: {elapsed_time} seconds\n")


                start_time = time.perf_counter()
                merged_df=data      #ADD THIS FOR NO SPATIAL SUBSAMPLING
                grouped = merged_df.groupby(['year', 'month', 'day', 'hour'])
                grouped_mean = grouped[['x', 'y']].mean()

               # Iterate through each row in the DataFrame
                for group_key, row in grouped:
                    longitude = grouped_mean.loc[group_key]['x']
                    latitude = grouped_mean.loc[group_key]['y']
                    row['time'] = pd.to_datetime(row['time'])
                    date_str = f"{group_key[0]:04d}-{group_key[1]:02d}-{group_key[2]:02d}-{group_key[3]:02d}"
                    local_time = get_local_time(latitude, longitude, date_str)
                    elevation, declination = get_solar_position(latitude, longitude, date_str)

                    merged_df.loc[row.index, 'DECL'] = declination*180/3.14159265

                    merged_df.loc[row.index, 'SOLAR_ELEV'] = elevation
                    merged_df.loc[row.index, 'local_time'] = local_time



                #ADD WIND DIRECTION
                merged_df['WD']=np.mod(180+np.rad2deg(np.arctan2( merged_df['U10'],  merged_df['V10'])),360)

                merged_df['T_2M_COR']=merged_df['T_2M_SL']-6.5*(merged_df['ELEV'])/1000


                process_city_data(city, year, month)

                end_time = time.perf_counter()
                elapsed_time = end_time - start_time
                print(f"Elapsed time for step adding solar features + export: {elapsed_time} seconds")
                with open('run_time2.txt', 'a') as f:
                    f.write(f"Elapsed time for adding solar features + export: {elapsed_time} seconds\n")









