"""
Calculate the Arctic Amplification (AA) difference of anomalies metric from
Francis & Vavrus (2015), Arnheim et al. (2025) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import numpy as np
import os
import time
import xarray as xr

def cosine_weighted_spatial_mean(data, coord=None):
    coord = coord or ('lat' if 'lat' in data.coords else 'latitude' if 'latitude' in data.coords else None)
    if not coord:
        raise ValueError("No 'lat' or 'latitude' coordinate found in the data array.")
    if 'lat' in data.coords and 'latitude' in data.coords:
        raise ValueError("Both 'lat' and 'latitude' coordinates exist.")
    weights = np.cos(np.deg2rad(data[coord]))
    return (data * weights).sum(dim=coord) / (data.notnull() * weights).sum(dim=coord)

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

def region_calculations(data, region, clim_start, clim_end):
    mean = cosine_weighted_spatial_mean(data.where(data.latitude >= region, drop=True))
    clim = mean.sel(year=slice(str(clim_start), str(clim_end))).mean(dim='year')
    anomalies = mean - clim
    return mean, anomalies

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(base_dir, 'Data')

### Climatology
clim_start, clim_end = (1981, 2010)

print(f'\n---- Arctic Amplification Metric (Difference of Anomalies) ----\n')

### Load data
filename = os.path.join(data_dir, 'ERA5_Global_1940-2025_T2M_Monthly.nc')
datat = xr.open_dataset(filename)['t2m']

### Process data
datat_annual = datat.groupby('time.year').mean(dim='time').mean(dim='longitude')

### Compute AA_1
Arctic_mean, Arctic_anom = region_calculations(datat_annual, 70, clim_start, clim_end)
NH_mean, NH_anom = region_calculations(datat_annual, 0, clim_start, clim_end)
AA_1 = Arctic_anom - NH_anom

### Create dataset
AA_1_dataset = xr.Dataset(
    {'AA_1': ('year', AA_1.values),
     'Arctic_anom': ('year', Arctic_anom.values),
     'NH_anom': ('year', NH_anom.values),
     'Arctic': ('year', Arctic_mean.values),
     'NH': ('year', NH_mean.values)},
    coords={'year': datat_annual.year})

### Save dataset
output_filename = os.path.join(data_dir, 'ERA5_AA_1_T2M_1940-2025_Annual.nc')
AA_1_dataset.to_netcdf(output_filename)
print(f'Saved {output_filename}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')