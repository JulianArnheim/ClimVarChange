"""
Calculate Zonal Index (ZON) from Woollings (2008), Peings et al.
(2017, 2018), Arnheim et al. (2025) from ERA5 reanalysis

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

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(base_dir, 'Data')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Zonal Index (ZON) ----\n')

for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)

    output_file_decade = os.path.join(data_dir, f'ERA5_ZON_Sectors_{start_year}-{end_year}_Daily.nc')
    if os.path.exists(output_file_decade):
        print(f'  Skipping {start_year}–{end_year}, file exists.')
        continue

    filename = os.path.join(data_dir, f'ERA5_NH_{start_year}-{end_year}_Z500_Daily.nc')
    if not os.path.exists(filename):
        print(f'  Data file missing: {filename}, skipping.')
        continue

    print(f'\n--- Processing Decade {start_year}–{end_year} ---\n')

    ### Load decade data
    dataz = xr.open_dataset(filename, chunks={'time': 10})['z'].sortby('time')

    ### Define longitude sectors using boolean masks
    longitudes = dataz['longitude']
    sectors = {'NH': (longitudes >= -180) & (longitudes < 180),
               'AM': (longitudes >= -140) & (longitudes <= -50),
               'AT': (longitudes >= -50) & (longitudes <= 40),
               'AS': (longitudes >= 40) & (longitudes <= 130),
               'PA': ((longitudes >= 130) | (longitudes <= -140))}

    ### Compute ZON for each sector and decade
    zonal_data = {}
    for key, mask in sectors.items():
        print(f'  Sector: {key}')

        ### Initializations
        values = np.empty(dataz.time.size)
        prev_year = None

        ### Loop through each day in the decade
        for t in range(dataz.time.size):

            ### Print progress whenever a new year
            if (curr_year := dataz.time.dt.year.values[t]) != prev_year:
                print(f'    Year: {curr_year}')
                prev_year = curr_year

            ### Compute difference between spatial mean geopotential heights for specific sector
            zonal_mean = dataz.isel(time=t).sel(longitude=longitudes[mask]).mean(dim='longitude')
            lat_name = 'lat' if 'lat' in dataz.dims else 'latitude'
            high = cosine_weighted_spatial_mean(zonal_mean.sel({lat_name: slice(90, 60)}), coord=lat_name)
            low = cosine_weighted_spatial_mean(zonal_mean.sel({lat_name: slice(50, 20)}), coord=lat_name)
            values[t] = low - high

        zonal_data[f'ZON_{key}'] = (['time'], values)

    ### Save ZON decade dataset
    ds_out = xr.Dataset(zonal_data, coords={'time': dataz.time.values})
    ds_out.to_netcdf(output_file_decade)
    print(f'  Saved {output_file_decade}, the ZON data\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')