"""
Preprocess the European Centre for Medium-Range Weather Forecasts (ECMWF) Reanalysis version 5 (ERA5)
monthly 500 hPa vertical pressure velocity (OMEGA500) dataset

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

### Preprocess a single file
base_filename = f'ERA5_Global_{year_min}-{year_max}_OMEGA500_Monthly'
input_file = os.path.join(data_dir, 'raw', f'{base_filename}_raw.nc')
output_file = os.path.join(data_dir, f'{base_filename}.nc')
if os.path.exists(input_file):
    ds = xr.open_dataset(input_file)
    print(f'Loaded: {input_file}')

    ### Rename valid_time to time
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})

    ### Drop unwanted coordinates
    for var in ['number', 'expver']:
        if var in ds.coords:
            ds = ds.drop_vars(var)

    ### Drop single-level pressure_level coordinate if present
    if 'pressure_level' in ds.coords:
        if ds.coords['pressure_level'].size == 1:
            ds = ds.drop_vars('pressure_level').squeeze('pressure_level', drop=True)

    ### Strip hour/minute/second from time
    ds = ds.assign_coords(time=ds.indexes['time'].normalize().values.astype('datetime64[D]'))

    ### Convert coordinates that are float64 to float32
    for coord in ds.coords:
        if ds[coord].dtype == 'float64':
            ds[coord] = ds[coord].astype('float32')

    ### Rename vertical velocity, kept in Pa s⁻¹ (negative values are rising motion)
    if 'w' in ds.data_vars:
        ds = ds.rename({'w': 'OMEGA500'})
        ds['OMEGA500'].attrs['units'] = 'Pa s-1'
        ds['OMEGA500'].attrs['long_name'] = 'Vertical pressure velocity at 500 hPa'

    ### Save
    ds.to_netcdf(output_file)
    print(f'Saved {output_file}\n{ds}\n')
else:
    print(f'Input file not found: {input_file}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')