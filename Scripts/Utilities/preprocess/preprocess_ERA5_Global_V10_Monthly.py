"""
Preprocess the European Centre for Medium-Range Weather Forecasts (ECMWF) Reanalysis version 5 (ERA5)
monthly 10-m meridional wind (V10) dataset

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

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Preprocess ERA5 ----\n')

### Preprocess a single file
base_filename = f'ERA5_Global_{year_min}-{year_max}_V10_Monthly'
input_file, output_file = os.path.join(data_dir, 'raw', f'{base_filename}_raw.nc'), os.path.join(data_dir, f'{base_filename}.nc')
if os.path.exists(input_file):
    ds = xr.open_dataset(input_file)
    print(f'Loaded: {input_file}')

    ### Rename valid_time to time, drop unwanted coordinates, and convert float64 to float32
    ds = ds.rename({'valid_time': 'time'}) if 'valid_time' in ds.coords else ds
    ds = ds.drop_vars([v for v in ['number', 'expver'] if v in ds.coords])
    ds = ds.assign_coords({c: ds[c].astype('float32') for c in ds.coords if ds[c].dtype == 'float64'})

    ### Save
    ds.to_netcdf(output_file)
    print(f'Saved {output_file}\n{ds}\n')
else:
    print(f'Input file not found: {input_file}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')