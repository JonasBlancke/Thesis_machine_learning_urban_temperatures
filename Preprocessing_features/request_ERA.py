import cdsapi

c = cdsapi.Client()

year = 2009  # Set the year to 2009

for month in range(8, 13):  # Loop over months 08 to 12
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                'boundary_layer_height', 'convective_available_potential_energy', 'k_index',
                'surface_net_solar_radiation', 'surface_net_thermal_radiation', 'surface_pressure',
                'surface_sensible_heat_flux', 'total_cloud_cover', 'total_column_water',
                'total_precipitation',
            ],
            'year': str(year),
            'month': "{month:02d}".format(month=month),
            'day': '01',  # Set day to 01 to download data for the first day of each month
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area': [
                72, -25, 35, 65,
            ],
        },
        'ERA5_{year}_{month:02d}.nc'.format(year=year, month=month)
    )
