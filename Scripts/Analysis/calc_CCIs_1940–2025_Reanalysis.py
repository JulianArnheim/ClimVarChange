"""
Calculate Climate Change Indices (CCIs) from Peings et al.
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

def level_weighted_mean(data):
    dp = xr.DataArray(np.abs(np.gradient(data['level'].values.astype(float))), coords={'level': data['level']}, dims='level')  # layer thickness [hPa]
    return data.weighted(dp).mean(dim='level')

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(base_dir, 'Data')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Climate Change Indices (CCIs) ----\n')

### Month and sector options
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}
longitudes_options = {'1. NH (180°W–180°E)': lambda lon: (lon >= 0) & (lon < 360),
                      '2. AM (140°W–50°W)': lambda lon: (lon >= 360 - 140) & (lon < 360 - 50),  # -140 to -50
                      '3. AT (50°W–40°E)': lambda lon: ((lon >= 360 - 50) & (lon < 360)) | ((lon >= 0) & (lon <= 40)),  # -50 to 40
                      '4. AS (40°E–130°E)': lambda lon: (lon >= 40) & (lon <= 130),  # 40 to 130
                      '5. PA (130°E–140°W)': lambda lon: (lon >= 130) & (lon < 360 - 140)}  # 130 to -140

#################################################
##### Step 1: Precompute annual zonal means #####
#################################################
for sel_month, mons in months_options.items():
    for sel_longitude, lon_func in longitudes_options.items():

        ### String corrections
        text_months, text_longitudes = sel_month.split()[1], sel_longitude.split()[1]

        print(f'{text_months} | {text_longitudes}')

        ### Load annual zonal mean temperature profile decade files
        for start_year in range(year_min, 2030, 10):

            ### Define input and output year ranges
            data_start_year = start_year
            end_year = min(start_year + 9, year_max)

            ### DJF 1940 cannot be computed because December 1939 is unavailable
            output_start_year = year_min + 1 if text_months == 'DJF' and start_year == year_min else start_year

            ### Annual zonal mean file if available
            zonal_file = os.path.join(data_dir, f'ERA5_Global_{output_start_year}-{end_year}_TProfile_{text_months}_{text_longitudes}_AnnualZonalMeans.nc')

            ### Skip calculation if the precomputed file already exists
            if os.path.exists(zonal_file):
                print(f'  Exists: {zonal_file}, skipping.')
                continue

            ### Load the corresponding monthly temperature profile file
            monthly_file = os.path.join(data_dir, f'ERA5_Global_{data_start_year}-{end_year}_TProfile_Monthly.nc')

            ### Skip decade if the required monthly file is unavailable
            if not os.path.exists(monthly_file):
                print(f'  Missing {monthly_file}, skipping {data_start_year}–{end_year}')
                continue

            ### Load monthly temperature profile data
            ds = xr.open_dataset(monthly_file, chunks='auto', engine='netcdf4')
            lon_coord = 'longitude' if 'longitude' in ds.coords else 'lon'

            ### Compute annual zonal mean for select months and longitudes
            ds = ds.sel(time=ds['time.month'].isin(mons), **{lon_coord: lon_func(ds[lon_coord])})

            ### Winter season handling that assigns December to the following year and computes seasonal means accordingly
            if text_months == 'DJF':

                ### Append December from previous year if available
                if start_year > year_min:
                    prev_file = os.path.join(data_dir, f'ERA5_Global_{start_year - 10}-{start_year - 1}_TProfile_Monthly.nc')
                    if os.path.exists(prev_file):
                        prev_ds = xr.open_dataset(prev_file, chunks='auto', engine='netcdf4')
                        prev_ds = prev_ds.sel(time=prev_ds['time.month'] == 12, **{lon_coord: lon_func(prev_ds[lon_coord])})
                        ds = xr.concat([prev_ds, ds], dim='time')

                ### Shift December to following seasonal year
                season_year = ds.time.dt.year + (ds.time.dt.month == 12)
                ds = ds.assign_coords(season_year=('time', season_year.data))

                ### Compute seasonal zonal mean profiles if all three months are available
                monthly_zonal = ds.mean(dim=lon_coord)
                month_count = monthly_zonal['time.month'].groupby(monthly_zonal['season_year']).count()
                annual = monthly_zonal.groupby('season_year').mean(dim='time').rename({'season_year': 'year'})
                valid_years = month_count.where(month_count == 3, drop=True).season_year
                annual = annual.where(annual.year.isin(valid_years), drop=True)

            else:
                annual = ds.mean(dim=lon_coord).groupby('time.year').mean(dim='time')

            ### Select years and sort
            annual = annual.where((annual.year >= output_start_year) & (annual.year <= end_year), drop=True).sortby('year')

            ### Save
            annual.to_netcdf(zonal_file)
            print(f'  Saved {zonal_file}')

#################################################
##### Step 2: Compute CCIs from zonal means #####
#################################################
CCIs = xr.Dataset()
for sel_month, mons in months_options.items():
    for sel_longitude, lon_func in longitudes_options.items():

        ### String corrections
        text_months, text_longitudes = sel_month.split()[1], sel_longitude.split()[1]

        ### Load all decades
        decadal_means = []
        for start_year in range(year_min, 2030, 10):

            ### Define decade range
            end_year = min(start_year + 9, year_max)

            ### DJF begins in 1941 because December 1939 is unavailable
            start_year_file = year_min + 1 if text_months == 'DJF' and start_year == year_min else start_year

            ### Annual zonal mean file if available
            zonal_file = os.path.join(data_dir, f'ERA5_Global_{start_year_file}-{end_year}_TProfile_{text_months}_{text_longitudes}_AnnualZonalMeans.nc')

            ### Add available decade to list for concatenation
            if os.path.exists(zonal_file):
                decadal_means.append(xr.open_dataset(zonal_file))

        ### Skip season/sector combination if no data are available
        if not decadal_means:
            print(f'  No data for {text_months} {text_longitudes}, skipping...')
            continue

        ### Concatenate decadal means into a single dataset
        mean_profile = xr.concat(decadal_means, dim='year').sortby('year').drop_duplicates('year')

        ### Compute CCIs
        for region in ['UTW', 'AA', 'PST', 'UPTG', 'LOTG']:
            ### Upper troposphere warming (UTW)
            if region == 'UTW':
                region_data = mean_profile.sel(latitude=slice(20, -20), level=slice(400, 150))
                region_data = level_weighted_mean(cosine_weighted_spatial_mean(region_data))
            ### Arctic amplification (AA)
            elif region == 'AA':
                region_data = mean_profile.sel(latitude=slice(90, 60), level=slice(1000, 700))
                region_data = level_weighted_mean(cosine_weighted_spatial_mean(region_data))
            ### Polar stratospheric temperature (PST)
            elif region == 'PST':
                region_data = mean_profile.sel(latitude=slice(90, 65), level=slice(250, 30))
                region_data = level_weighted_mean(cosine_weighted_spatial_mean(region_data))
            ### Upper troposphere temperature gradient (UPTG)
            elif region == 'UPTG':
                u_trop = mean_profile.sel(latitude=slice(20, -20), level=slice(400, 150))
                u_arctic = mean_profile.sel(latitude=slice(90, 65), level=slice(400, 150))
                u_trop_mean = level_weighted_mean(cosine_weighted_spatial_mean(u_trop))
                u_arctic_mean = level_weighted_mean(cosine_weighted_spatial_mean(u_arctic))
                region_data = u_trop_mean - u_arctic_mean
            ### Lower troposphere temperature gradient (LOTG)
            elif region == 'LOTG':
                l_trop, l_arctic = mean_profile.sel(latitude=slice(20, -20), level=slice(1000, 700)), mean_profile.sel(latitude=slice(90, 65), level=slice(1000, 700))
                l_trop_mean = level_weighted_mean(cosine_weighted_spatial_mean(l_trop))
                l_arctic_mean = level_weighted_mean(cosine_weighted_spatial_mean(l_arctic))
                region_data = l_trop_mean - l_arctic_mean
            CCIs[f'ERA5_{text_months}_{text_longitudes}_{region}_CCIs'] = region_data['t']

### Save dataset
output_filename = os.path.join(data_dir, f'ERA5_CCIs_{year_min}-{year_max}_Annual.nc')
CCIs.to_netcdf(output_filename)
print(f'\nSaved {output_filename}\n{CCIs}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')