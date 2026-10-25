"""
Retrieve the European Centre for Medium-Range Weather Forecasts (ECMWF) Reanalysis version 5 (ERA5)
monthly large-scale precipitation (PRECL) using the Climate Data Store (CDS) Application Programming Interface (API)

Notes
-----
    Author : Julian B Arnheim
"""

### Import modules
import cdsapi
import os
import time

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directory
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data', 'raw'))

### Variable
var = 'PRECL'

### Dataset
ds = 'reanalysis-era5-single-levels-monthly-means'

### Years and months
year_min, year_max = 1940, 2025
years, months = [str(y) for y in range(year_min, year_max + 1)], [f'{m:02d}' for m in range(1, 13)]

print(f'\n---- Download ERA5 Monthly Large-Scale Precipitation (PRECL) ----\n')

### Save
output_file = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_{var}_Monthly_raw.nc')
print(f'Requesting all years {year_min}–{year_max}, monthly means, globally... Output: {output_file}\n')
cdsapi.Client().retrieve(ds, {
    'product_type': 'monthly_averaged_reanalysis',
    'variable': 'large_scale_precipitation',
    'year': years,
    'month': months,
    'time': '00:00',  # monthly mean uses a single time
    'format': 'netcdf',
}, output_file)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')