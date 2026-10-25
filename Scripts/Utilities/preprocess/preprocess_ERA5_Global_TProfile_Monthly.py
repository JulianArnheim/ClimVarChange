"""
Preprocess the European Centre for Medium-Range Weather Forecasts (ECMWF) Reanalysis version 5 (ERA5)
monthly Temperature (T) profile dataset

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import os
import time
import xarray as xr

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Preprocess ERA5 ----\n')

### Loop through decade/partial-decade files and preprocess individually
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)

    ### Files
    base_filename = f'ERA5_Global_{start_year}-{end_year}_TProfile_Monthly'
    input_file = os.path.join(data_dir, 'raw', f'{base_filename}_raw.nc')
    output_file = os.path.join(data_dir, f'{base_filename}.nc')

    if os.path.exists(input_file):
        ds = xr.open_mfdataset(input_file, combine='by_coords', engine='netcdf4', chunks='auto')

        print(f'Loaded: {start_year}–{end_year}')

        ### Rename valid_time to time
        if 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})

        ### Rename pressure_level to level
        if 'pressure_level' in ds.coords:
            ds = ds.rename({'pressure_level': 'level'})

        ### Drop unwanted coordinates if they exist
        for coord in ['number', 'expver']:
            if coord in ds.coords:
                ds = ds.drop_vars(coord)

        ### Convert coordinates that are float64 to float32
        for coord in ds.coords:
            if ds[coord].dtype == 'float64':
                ds[coord] = ds[coord].astype('float32')

        ### Convert from Kelvin to degrees Celsius
        if 't' in ds.data_vars:
            ds['t'] = ds['t'] - 273.15
            ds['t'].attrs['units'] = '°C'

        ### Save
        ds.to_netcdf(output_file)
        print(f'Saved {output_file}')
        print(f'{ds}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')