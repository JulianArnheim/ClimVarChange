"""
Preprocess the Pan-Arctic Ice Modeling and Assimilation System (PIOMAS) assimilative ice-ocean model
monthly Sea Ice Volume (SIV) from the University of Washington Polar Science Center

Notes
-----
    Data : https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/
    Author : Julian B Arnheim
"""

### Import modules
import numpy as np
import os
import pandas as pd
import time
import xarray as xr

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directory
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data'))

print(f'\n---- Preprocess PIOMAS SIV ----\n')

### Variable
var = 'SIV'

### Years
year_min, year_max = 1979, 2025

### Files
input_file = os.path.join(data_dir, 'raw', f'PIOMAS_Arctic_{year_min}-{year_max}_{var}_Monthly_raw.csv')
output_file = os.path.join(data_dir, f'PIOMAS_Arctic_{year_min}-{year_max}_{var}_Monthly.nc')

### Preprocess
if os.path.exists(output_file):
    print(f'File exists: {output_file}\n')
elif os.path.exists(input_file):
    print(f'File being made: {output_file}')

    ### Load raw year x month table
    df = pd.read_csv(input_file)
    print(f'Loaded: {input_file}')

    ### Reshape from year x month table to a monthly time series
    df = df.melt(id_vars='year', var_name='month', value_name=var)
    df['time'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-' + df['month'], format='%Y-%b')
    siv = df.sort_values('time').set_index('time')[var]

    ### Convert missing values (-1) to NaN and subset years
    siv = siv.where(siv >= 0, np.nan).loc[f'{year_min}-01-01': f'{year_max}-12-31']
    n_months = 12 * (year_max - year_min + 1)
    if siv.size != n_months:
        raise ValueError(f'Unexpected number of months: {siv.size}, expected {n_months}')
    print(f'Time range: {siv.index.min():%Y-%m} to {siv.index.max():%Y-%m}, missing months: {int(siv.isna().sum())}')

    ### Create xarray Dataset
    ds = xr.Dataset({var: ('time', siv.values)}, coords={'time': siv.index.values})
    ds[var].attrs = {'long_name': 'Sea ice volume integrated over the PIOMAS domain', 'units': '10^3 km^3'}
    ds.attrs = {'source': 'PIOMAS v2.1, University of Washington Polar Science Center',
                'citation': 'Schweiger et al. (2011), J. Geophys. Res., doi:10.1029/2011JC007084'}

    ### Save
    ds.to_netcdf(output_file)
    print(f'Saved {output_file}\n{ds}\n')
else:
    print(f'Input file not found: {input_file}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')