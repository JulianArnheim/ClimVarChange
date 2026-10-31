"""
Calculate Blocking Index (BLO) from Tibaldi & Molteni (1990), Scherrer et al. (2006),
Gong & Luo (2017), Peings et al. (2017, 2018), Arnheim et al. (2025) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import glob
import numpy as np
import os
import time
import xarray as xr

def compute_sector_BLO(BLO_5d, sector_masks, collapse_lat=True):
    BLO_sector = {}
    for key, mask in sector_masks.items():
        ### Sector specification
        BLO_sel = BLO_5d.sel(longitude=BLO_5d.longitude[mask])
        ### Collapse into int8 (1 = blocking, 0 = no blocking) per time step
        BLO_any = BLO_sel.any(dim=('latitude', 'longitude') if collapse_lat else 'longitude')
        BLO_sector[f'BLO_{key}'] = BLO_any.astype(np.int8)
    return BLO_sector

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

### Latitudes
lats = np.arange(35, 76, 2.5)

print(f'\n---- Blocking Index (BLO) ----\n')

#####################################
##### Step 1: Compute 1-day BLO #####
#####################################
print(f'Computing 1-day BLO files by decade...\n')

for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)

    outfile_BLO_1d = os.path.join(data_dir, f'ERA5_BLO_1d_{start_year}-{end_year}_Daily.nc')
    if os.path.exists(outfile_BLO_1d):
        print(f'Skipping {start_year}–{end_year}, 1-day BLO file exists.')
        continue

    filename = os.path.join(data_dir, f'ERA5_NH_{start_year}-{end_year}_Z500_Daily.nc')
    if not os.path.exists(filename):
        print(f'Data file missing: {filename}, skipping.')
        continue

    print(f'Loaded: {start_year}–{end_year}')

    ### Load data with optional chunking
    dataz_decade = xr.open_dataset(filename, chunks='auto').chunk({'time': 30})

    ### Process data
    dataz_decade = dataz_decade.sortby('time').sel(latitude=slice(20, None, -1))
    dataz_decade = dataz_decade.sel(longitude=np.arange(-180, 181, 2.5), method='nearest')

    ### Precompute latitude indices
    lat_vals = dataz_decade.latitude.values
    lat_N_idx = np.array([np.abs(lat_vals - (lat + 15)).argmin() for lat in lats])
    lat_0_idx = np.array([np.abs(lat_vals - lat).argmin() for lat in lats])
    lat_S_idx = np.array([np.abs(lat_vals - (lat - 15)).argmin() for lat in lats])

    ### Compute 1-day blocking vectorized
    Z = dataz_decade['z']
    GH_N = Z.isel(latitude=lat_N_idx).assign_coords(latitude=lats)
    GH_0 = Z.isel(latitude=lat_0_idx).assign_coords(latitude=lats)
    GH_S = Z.isel(latitude=lat_S_idx).assign_coords(latitude=lats)

    ### Geopotential height gradients
    GHGN = (GH_N - GH_0) / 15.0
    GHGS = (GH_0 - GH_S) / 15.0

    ### Blocking criterion
    BLO_1d = ((GHGN < -10) & (GHGS > 0)).astype(np.int8).rename('BLO')
    BLO_1d = BLO_1d.assign_coords(time=dataz_decade.time, longitude=dataz_decade.longitude)

    ### Save 1-day BLO
    BLO_1d.to_netcdf(outfile_BLO_1d)
    print(f'  Saved {outfile_BLO_1d}, the 1-day BLO')

#################################################
##### Step 2: Compute 5-day persistence BLO #####
#################################################
print(f'\nComputing 5-day persistence BLO files by decade...\n')

BLO_files = sorted(glob.glob(os.path.join(data_dir, 'ERA5_BLO_1d_*.nc')))
if not BLO_files:
    raise ValueError('No 1-day BLO files found. Run Step 1 first.')

longitudes = xr.open_dataset(BLO_files[0]).longitude
sectors = {'NH': (longitudes >= -180) & (longitudes < 180),
           'AM': (longitudes >= -140) & (longitudes <= -50),
           'AT': (longitudes >= -50) & (longitudes <= 40),
           'AS': (longitudes >= 40) & (longitudes <= 130),
           'PA': ((longitudes >= 130) & (longitudes <= 180)) | ((longitudes >= -180) & (longitudes <= -140))}

for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)

    ### Skip if output exists
    out_filename = os.path.join(data_dir, f'ERA5_BLO_5d_Sectors_{start_year}-{end_year}_Daily.nc')
    if os.path.exists(out_filename):
        print(f'Skipping {start_year}–{end_year}, file exists.')
        continue

    ### Load only relevant 1-day BLO files
    BLO_slices = []
    for f in BLO_files:
        file_start, file_end = map(int, os.path.basename(f).split('_')[-2].replace('.nc', '').split('-'))
        if file_end < start_year - 1 or file_start > end_year + 1:
            continue
        ds = xr.open_dataset(f)['BLO']
        ds = ds.sel(time=slice(f'{start_year-10}-12-31', f'{end_year+10}-01-01'))
        BLO_slices.append(ds)

    ### Apply rolling persistence, flagging every day of each run of at least five consecutive blocked days
    BLO_1d_overlap = xr.concat(BLO_slices, dim='time').sortby('time')
    window_end = (BLO_1d_overlap.rolling(time=5, min_periods=5).sum() >= 5).astype(np.int8)  # 1 on the last day of each fully blocked 5-day window
    BLO_5d = (window_end.rolling(time=5, min_periods=1).max().shift(time=-4, fill_value=0) >= 1).astype(np.int8)  # extend the flag back over the window

    ### Trim decade
    BLO_5d = BLO_5d.sel(time=slice(f'{start_year}-01-01', f'{end_year}-12-31'))

    ### Sector specification
    BLO_sector_data = compute_sector_BLO(BLO_5d, sectors, collapse_lat=True)

    ### Save 5-day BLO decade dataset
    BLO_ds = xr.Dataset(BLO_sector_data, coords={'time': BLO_5d.time})
    BLO_ds.to_netcdf(out_filename)
    print(f'  Saved {out_filename}, the 5-day BLO\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')