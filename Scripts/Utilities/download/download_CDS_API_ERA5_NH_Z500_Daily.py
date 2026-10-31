"""
Retrieve the European Centre for Medium-Range Weather Forecasts (ECMWF) Reanalysis version 5 (ERA5)
daily 500 hPa geopotential height (Z500) using the Climate Data Store (CDS) Application Programming Interface (API)

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

### Months, days, years, and 10-year chunks
months = [f'{m:02d}' for m in range(1, 13)]
days = [f'{d:02d}' for d in range(1, 32)]  # CDS ignores invalid days, leap years handled automatically
year_min, year_max = 1940, 2025
decade_chunks = [range(s, min(s + 10, year_max + 1)) for s in range(year_min, year_max + 1, 10)]

print(f'\n---- Download ERA5 Daily Northern Hemisphere 500 hPa Geopotential Height (Z500) ----\n')

### Save in decades
c = cdsapi.Client()
for chunk in decade_chunks:
    years_list = [str(y) for y in chunk]
    output_file = os.path.join(data_dir, f'ERA5_NH_{years_list[0]}-{years_list[-1]}_Z500_Daily_raw.nc')
    print(f'Requesting years {years_list[0]}–{years_list[-1]}... Output: {output_file}\n')
    c.retrieve(
        'reanalysis-era5-pressure-levels', {
        'product_type': 'reanalysis', 'variable': 'geopotential',
        'pressure_level': '500',
        'year': years_list, 'month': months, 'day': days,
        'time': ['12:00'],  # daily 12 UTC
        'area': [90, -180, 0, 180],  # NH only
        'format': 'netcdf',
    }, output_file)

### End time
print(f'\n---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')