"""
Calculate Jet Latitude Index (JLI) and Jet Speed Index (JSI) from
White et al. (2019) or Bracegirdle (2022) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from scipy.interpolate import CubicSpline
import numpy as np
import os
import pandas as pd
import time
import xarray as xr

def compute_JLI_JSI(u_day, band_width=1):
    ### Latitude of maximum daily zonal wind
    max_lat = u_day['latitude'].isel(latitude=u_day.argmax(dim='latitude')).values
    lower_bound, upper_bound = max_lat - band_width / 2, max_lat + band_width / 2
    u_band = u_day.sel(latitude=slice(lower_bound, upper_bound))
    ### Skip if too few latitude points
    if u_band.sizes['latitude'] < 3:
        return np.nan, np.nan
    ### Cubic spline interpolation
    lat_interp = np.linspace(float(u_band.latitude.min()), float(u_band.latitude.max()), 10000)
    u_interp = CubicSpline(u_band.latitude, u_band.values)(lat_interp)
    ### Return JLI and JSI
    max_index = np.argmax(u_interp)
    return lat_interp[max_index], u_interp[max_index]

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

def load_level_data(lev):
    JLI_list, JSI_list, time_list = [], [], []
    ### Loop through decades
    for start_year in range(year_min, 2030, 10):
        end_year = min(start_year + 9, year_max)
        input_file = os.path.join(data_dir, f'ERA5_NA_{start_year}-{end_year}_U{lev}_Daily.nc')
        ### Notify if file is missing and skip
        if not os.path.exists(input_file):
            print(f'Missing: {input_file}')
            continue
        print(f'  Loaded: {start_year}–{end_year}')
        with xr.open_dataset(input_file) as ds:
            ### Compute zonal mean zonal wind
            datau = ds['u'].mean(dim='longitude').sel(latitude=slice(None, None, -1)).sortby('time')
            ### Loop through time steps
            for t in datau.time:
                ### Compute JLI and JSI
                JLI, JSI = compute_JLI_JSI(datau.sel(time=t))
                ### Append
                JLI_list.append(JLI)
                JSI_list.append(JSI)
                time_list.append(pd.Timestamp(t.values))
    return (np.array(JLI_list), np.array(JSI_list), np.array(time_list, dtype='datetime64[ns]'))

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(base_dir, 'Data')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Jet Latitude Index (JLI) and Jet Speed Index (JSI) ----\n')

### Output filename
out_filename = os.path.join(data_dir, f'ERA5_NA_JLI_JSI_{year_min}-{year_max}_Daily.nc')

### Build dataset by looping through pressure levels
data_vars, time_coord = {}, None
levels = [700, 750, 800, 850, 900, 925]
for lev in levels:
    print(f'Processing level: {lev} hPa')

    ### Load JLI and JSI level data
    JLI_array, JSI_array, time_array = load_level_data(lev)

    ### Save time coordinate once
    if time_coord is None:
        time_coord = time_array

    ### Build data variables
    data_vars[f'JLI_U{lev}'] = (('time',), JLI_array, {'long_name': f'Jet Latitude Index at {lev} hPa', 'units': 'degrees_north'})
    data_vars[f'JSI_U{lev}'] = (('time',), JSI_array, {'long_name': f'Jet Speed Index at {lev} hPa', 'units': 'm s-1'})

### Save dataset
ds_out = xr.Dataset(data_vars=data_vars, coords={'time': time_coord})
ds_out.to_netcdf(out_filename)
print(f'Saved {out_filename}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')