"""
Preprocess the European Centre for Medium-Range Weather Forecasts (ECMWF) Reanalysis version 5 (ERA5)
monthly 2-m Air Temperature (T2M) dataset

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

### Directory
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data'))

print(f'\n---- Preprocess ERA5 ----\n')

### Preprocess a single file
input_file = os.path.join(data_dir, 'raw', f'ERA5_Global_1940-2025_T2M_Monthly_raw.nc')
output_file = os.path.join(data_dir, f'ERA5_Global_1940-2025_T2M_Monthly.nc')
if os.path.exists(input_file):
    ds = xr.open_dataset(input_file)
    print(f'Loaded: {input_file}')

    ### Rename valid_time to time, drop unwanted coordinates, and convert from Kelvin to degrees Celsius
    ds = ds.rename({'valid_time': 'time'}) if 'valid_time' in ds.coords else ds
    ds = ds.drop_vars([c for c in ['number', 'expver'] if c in ds.coords])
    ds = ds.assign_coords({c: ds[c].astype('float32') for c in ds.coords if ds[c].dtype == 'float64'})
    if 't2m' in ds.data_vars:
        ds['t2m'] = ds['t2m'] - 273.15
        ds['t2m'].attrs['units'] = '°C'

    ### Save
    ds.to_netcdf(output_file)
    print(f'Saved {output_file}\n{ds}\n')
else:
    print(f'Input file not found: {input_file}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')