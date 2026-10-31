"""
Preprocess the monthly Oceanic Niño Index (ONI) from the NOAA Physical Sciences Laboratory (PSL) and the monthly
Relative Oceanic Niño Index (RONI) from the NOAA Climate Prediction Center (CPC) into one NetCDF file

Notes
-----
    Data : https://psl.noaa.gov/data/correlation/oni.data
    Data : https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt
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

### Years and files
year_min, year_max = 1950, 2026
oni_file = os.path.join(data_dir, 'raw', f'PSL_Nino34_{year_min}-{year_max}_ONI_Monthly_raw.txt')
roni_file = os.path.join(data_dir, 'raw', f'CPC_Nino34_{year_min}-{year_max}_RONI_Monthly_raw.txt')
output_file = os.path.join(data_dir, f'NOAA_Nino34_{year_min}-{year_max}_ONI_RONI_Monthly.nc')

print(f'\n---- Preprocess ONI and RONI ----\n')

### Monthly time axis, each three-month season assigned to its center month (DJF -> January)
months = pd.date_range(f'{year_min}-01-01', f'{year_max}-12-01', freq='MS')
seasons = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']

### Load ONI, a year x month table with a "first last" year header and a missing-value footer
with open(oni_file, 'r') as f:
    lines = f.read().splitlines()
oni_years = range(int(lines[0].split()[0]), int(lines[0].split()[1]) + 1)
oni_rows = [line.split() for line in lines[1:] if len(line.split()) == 13 and line.split()[0].isdigit()]
oni_rows = [row for row in oni_rows if int(row[0]) in oni_years]
oni = pd.Series([float(v) for row in oni_rows for v in row[1:]],
                index=pd.to_datetime([f'{row[0]}-{m:02d}-01' for row in oni_rows for m in range(1, 13)]))
print(f'Loaded: {oni_file}')

### Load RONI, one season per row with columns SEAS, YR, ANOM
df = pd.read_csv(roni_file, sep=r'\s+')
df['month'] = df['SEAS'].map({season: i + 1 for i, season in enumerate(seasons)})
roni = pd.Series(df['ANOM'].values, index=pd.to_datetime(dict(year=df['YR'], month=df['month'], day=1)))
print(f'Loaded: {roni_file}')

### Convert missing values (-99.9) to NaN and place both indices on the monthly time axis
oni, roni = oni.where(oni > -99).reindex(months), roni.where(roni > -99).reindex(months)
for name, index in [('ONI', oni), ('RONI', roni)]:
    last = index.last_valid_index()
    print(f'{name}: {index.first_valid_index():%Y-%m} to {last:%Y-%m} ({seasons[last.month - 1]} {last.year})')

### Create xarray Dataset
ds = xr.Dataset({'ONI': ('time', oni.values.astype(np.float32)), 'RONI': ('time', roni.values.astype(np.float32))},
                coords={'time': months.values})
ds['ONI'].attrs = {'long_name': 'Oceanic Nino Index, 3-month running mean Nino 3.4 SST anomaly', 'units': 'degC'}
ds['RONI'].attrs = {'long_name': 'Relative Oceanic Nino Index, 3-month running mean relative Nino 3.4 SST anomaly', 'units': 'degC'}
ds.attrs = {'source': 'ONI from NOAA PSL (ERSSTv6), RONI from NOAA CPC (ERSSTv5)', 'note': 'Seasons assigned to their center month (DJF -> January)'}

### Save
ds.to_netcdf(output_file)
print(f'Saved {output_file}\n{ds}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')