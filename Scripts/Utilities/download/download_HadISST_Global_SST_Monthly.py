"""
Retrieve monthly Hadley Centre Sea Ice and Sea Surface Temperature (HadISST) data from the Met Office Hadley Centre

Notes
-----
    Data : https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz
    Author : Julian B Arnheim
"""

### Import modules
import gzip
import os
import pandas as pd
import requests
import shutil
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
year_min, year_max = 1870, 2025

### HadISST SST URL and file paths
url = 'https://www.metoffice.gov.uk/hadobs/hadisst/data/HadISST_sst.nc.gz'
gz_path = os.path.join(data_dir, 'HadISST_sst.nc.gz')
nc_path = gz_path[:-3]
output_path = os.path.join(data_dir, f'HadISST_Global_{year_min}-{year_max}_SST_Monthly.nc')

print(f'\n---- Download HadISST Monthly Sea Surface Temperature (SST) ----\n')

### Download
print(f'Downloading HadISST monthly SST ({year_min}–{year_max}) ...\n')
with requests.get(url, stream=True) as response:
    response.raise_for_status()
    with open(gz_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

### Decompress NetCDF file
with gzip.open(gz_path, 'rb') as f_in, open(nc_path, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)

### Subset and save
with xr.open_dataset(nc_path) as ds:

    ### Subset
    ds = ds.sel(time=slice(f'{year_min}-01-01', f'{year_max}-12-31'))

    ### Adjust coordinates and replace the HadISST sea-ice flag (-1000 °C) with the seawater freezing point (-1.8 °C)
    ds = ds.drop_vars('time_bnds').assign_coords(time=pd.date_range(f'{year_min}-01-01', periods=ds.sizes['time'], freq='MS'), longitude=(ds.longitude + 360) % 360)
    ds['sst'] = ds['sst'].where((ds['sst'] > -999) | ds['sst'].isnull(), -1.8)

    ### Reorder dimensions
    ds = ds.sortby('latitude').sortby('longitude').transpose('time', 'latitude', 'longitude').load()

    ### Save to NetCDF
    ds.to_netcdf(output_path)
    print(f'Saved {output_path}\n')

### Remove temporary files
for path in (gz_path, nc_path):
    os.remove(path)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')