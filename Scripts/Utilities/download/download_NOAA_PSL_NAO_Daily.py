"""
Retrieve the daily North Atlantic Oscillation (NAO) index from NOAA PSL, computed from NCEP/NCAR Reanalysis 1

Notes
-----
    Data : https://psl.noaa.gov/data/timeseries/daily/NAO/
    Author : Julian B Arnheim
"""

### Import modules
import os
import pandas as pd
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
year_min, year_max = 1948, 2025

### NAO index URL
url = 'https://downloads.psl.noaa.gov/Public/map/teleconnections/nao.reanalysis.t10trunc.1948-present.txt'

print(f'\n---- Download PSL Daily NAO Index ----\n')

print(f'Retrieving NAO index ({year_min}–{year_max}) ...\n')

### Download NAO index
df = pd.read_csv(url, sep=r'\s+', comment='#', header=None, names=['year', 'month', 'day', 'NAO'])
df['time'] = pd.to_datetime(df[['year', 'month', 'day']])

### Slice to desired years
df = df[df['time'].between(f'{year_min}-01-01', f'{year_max}-12-31')]

### Construct and save dataset
output_filename = os.path.join(data_dir, f'PSL_NAO_{year_min}-{year_max}_Daily.nc')
xr.Dataset({'NAO': ('time', df['NAO'])}, coords={'time': df['time']}).to_netcdf(output_filename)
print(f'Saved {output_filename}\n')

### End time
print(f'\n---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')