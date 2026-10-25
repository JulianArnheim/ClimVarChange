"""
Retrieve the European Centre for Medium-Range Weather Forecasts (ECMWF) Reanalysis version 5 (ERA5)
monthly Meridional Wind (V) using the Climate Data Store (CDS) Application Programming Interface (API)

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

### Dataset, years, months, and 10-year chunks
ds = 'reanalysis-era5-pressure-levels-monthly-means'
year_min, year_max = 1940, 2025
months = [f'{m:02d}' for m in range(1, 13)]
decade_chunks = [range(s, min(s + 10, year_max + 1)) for s in range(year_min, year_max + 1, 10)]

print(f'\n---- Download ERA5 Monthly Meridional Wind Profile (VProfile) ----\n')

### Save in decades
c = cdsapi.Client()
for chunk in decade_chunks:
    years_list = [str(y) for y in chunk]
    output_file = os.path.join(data_dir, f'ERA5_Global_{years_list[0]}-{years_list[-1]}_VProfile_Monthly_raw.nc')
    print(f'Requesting years {years_list[0]}–{years_list[-1]}, monthly '
          f'means, globally, all pressure levels... Output: {output_file}\n')
    c.retrieve(ds, {
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'v_component_of_wind',
        'pressure_level': [  '1',   '2',   '3',   '5',   '7',  '10',  '20',  '30',  '50',  '70', '100',
                           '125', '150', '175', '200', '225', '250', '300', '350', '400', '450', '500',
                           '550', '600', '650', '700', '750', '775', '800', '825', '850', '875', '900',
                           '925', '950', '975', '1000'],
        'year': years_list, 'month': months, 'time': '00:00', 'format': 'netcdf'}, output_file)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')