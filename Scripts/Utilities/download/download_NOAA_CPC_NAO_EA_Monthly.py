"""
Retrieve monthly North Atlantic Oscillation (NAO) and East Atlantic (EA)
teleconnection pattern indices from the NOAA Climate Prediction Center (CPC)

Notes
-----
    Data : https://www.cpc.ncep.noaa.gov/data/teledoc/telecontents.shtml
    Author : Julian B Arnheim
"""

### Import modules
from urllib.request import urlopen
import numpy as np
import os
import time
import xarray as xr

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

def load_CPC_index(url, name):
    """Load monthly CPC teleconnection index from YEAR, MONTH, INDEX columns."""

    ### Download and extract numeric data rows
    with urlopen(url) as response:
        data = []
        for line in response.read().decode('utf-8').splitlines():
            try:
                year, month, value = line.split()[:3]
                data.append([int(year), int(month), float(value)])
            except ValueError:
                continue

    ### Extract data, create monthly datetime coordinate, and convert missing values to NaN
    years, months, values = np.array(data).T
    times = np.array([np.datetime64(f'{int(y)}-{int(m):02d}') for y, m in zip(years, months)])
    values[values <= -90] = np.nan

    ### Return as xarray DataArray
    index_values = xr.DataArray(values, coords={'time': times}, dims='time', name=name)
    return index_values

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data')

### Years
year_min, year_max = 1950, 2025

### CPC data locations
NAO_site, EA_site = 'https://ftp.cpc.ncep.noaa.gov/wd52dg/data/indices/nao_index.tim', 'https://ftp.cpc.ncep.noaa.gov/wd52dg/data/indices/ea_index.tim'

print(f'\n---- Download CPC Monthly NAO and EA Indices ----\n')

### Download and process NAO index
print(f'Retrieving CPC NAO and EA monthly indices ({year_min}–{year_max}) ...')
NAO = load_CPC_index(NAO_site, 'NAO').sel(time=slice(f'{year_min}-01', f'{year_max}-12'))
print(f'Loaded NAO: {NAO.time.dt.year.min().item()}–{NAO.time.dt.year.max().item()}')

### Download and process EA index
EA = load_CPC_index(EA_site, 'EA').sel(time=slice(f'{year_min}-01', f'{year_max}-12'))
print(f'Loaded EA: {EA.time.dt.year.min().item()}–{EA.time.dt.year.max().item()}\n')

### Combine NAO and EA into a single xarray Dataset and save
teleconnections = xr.Dataset({'NAO': NAO, 'EA': EA})
output_filename = os.path.join(data_dir, f'CPC_NAO_EA_{year_min}-{year_max}_Monthly.nc')
teleconnections.to_netcdf(output_filename)
print(f'Saved {output_filename}\n')

### End time
print(f'\n---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')