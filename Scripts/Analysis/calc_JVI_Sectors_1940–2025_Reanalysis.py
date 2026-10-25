"""
Calculate Jet Variability Index (JVI) from Arnheim et al. (2025) from ERA5 reanalysis

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

def compute_monthly_jet_variability(monthly_data, lon_range):
    ### Load data for sector and latitude band
    dataz = monthly_data.sel(longitude=lon_range, latitude=slice(70, 30))
    ### Compute the zonal mean of each day
    daily_zonal_mean = dataz.mean(dim='longitude')
    daily_mean = cosine_weighted_spatial_mean(daily_zonal_mean, coord='latitude')
    ### Broadcast the 1D daily mean to the full 3D shape and binary mask where values are within ±5 m of spatial mean
    daily_mean_3d = daily_mean.broadcast_like(dataz)
    near_mean_mask = ((np.abs(dataz - daily_mean_3d)) <= 5).astype(int)
    ### Jet latitude cumulative distribution function (CDF) pooled over all days of the month
    monthly_mean_counts = near_mean_mask.sum(dim='longitude').mean(dim='time')
    norm_prob = monthly_mean_counts / monthly_mean_counts.sum()  # normalize
    sort_indices = np.argsort(norm_prob['latitude'].values)
    lats_sorted = norm_prob['latitude'].values[sort_indices]
    cum_prob = np.cumsum(norm_prob.values[sort_indices])
    ### Latitude range containing central 95% of the CDF
    high_thresh = .975
    low_thresh = .025
    upper_lat = lats_sorted[np.argmax(cum_prob >= high_thresh)]
    lower_lat = lats_sorted[np.argmax(cum_prob >= low_thresh)]
    JVI = upper_lat - lower_lat
    return JVI

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

print(f'\n---- Jet Variability Index (JVI) ----\n')

for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)

    out_filename = os.path.join(data_dir, f'ERA5_JVI_Sectors_{start_year}-{end_year}_Daily.nc')
    if os.path.exists(out_filename):
        print(f'Skipping {start_year}–{end_year}, JVI file exists.')
        continue

    filename = os.path.join(data_dir, f'ERA5_NH_{start_year}-{end_year}_Z500_Daily.nc')
    if not os.path.exists(filename):
        print(f'Data file missing: {filename}, skipping.')
        continue

    print(f'Loaded: {start_year}–{end_year}')

    ### Load data
    dataz = xr.open_dataset(filename, chunks={'time': 10})['z']
    dataz = dataz.sortby('time').sel(latitude=slice(70, 30))

    ### Sector definitions
    longitudes = dataz['longitude']
    sectors = {'NH': (longitudes >= -180) & (longitudes < 180),
               'AM': (longitudes >= -140) & (longitudes <= -50),
               'AT': (longitudes >= -50) & (longitudes <= 40),
               'AS': (longitudes >= 40) & (longitudes <= 130),
               'PA': ((longitudes >= 130) & (longitudes <= 180)) | ((longitudes >= -180) & (longitudes <= -140))}

    ### Compute JVI for each sector and decade
    jet_variability_data = {}
    for key, mask in sectors.items():
        print(f'  Sector: {key}')

        ### Sector specification
        JVI_sector_data = dataz.sel(longitude=longitudes[mask])

        ### Loop through each year-month pair
        prev_year = None
        year_month_pairs = sorted(set((t.year, t.month) for t in dataz.time.to_index()))
        jet_variability_values = []
        for year, month in year_month_pairs:

            ### Print progress whenever a new year
            if year != prev_year:
                print(f'    Year {year}')
                prev_year = year

            monthly_data = JVI_sector_data.sel(time=(JVI_sector_data['time.year'] == year) & (JVI_sector_data['time.month'] == month))
            JVI = compute_monthly_jet_variability(monthly_data, monthly_data['longitude'])

            ### Replicate the monthly JVI value for each day in the month
            jet_variability_values.extend([JVI] * monthly_data.sizes['time'])

        jet_variability_data[f'JVI_{key}'] = (('time',), np.array(jet_variability_values))

    ### Save JVI decade dataset
    ds_out = xr.Dataset(jet_variability_data, coords={'time': dataz.time.values})
    ds_out.attrs['note'] = 'JVI is computed once per calendar month from daily Z500 and repeated on every day of that month'
    ds_out.to_netcdf(out_filename)
    print(f'  Saved {out_filename}, the JVI data\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')