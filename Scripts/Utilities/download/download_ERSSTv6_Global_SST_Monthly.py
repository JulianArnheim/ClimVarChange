"""
Retrieve monthly National Oceanic and Atmospheric Administration (NOAA) Extended Reconstructed
Sea Surface Temperature (ERSSTv6) data from the NOAA National Centers for Environmental Information (NCEI)

Notes
-----
    Data : https://www.ncei.noaa.gov/products/extended-reconstructed-sst
    Author : Julian B Arnheim
"""

### Import modules
import os
import requests
import time
import xarray as xr

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directory
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data'))

### Years
year_min, year_max = 1850, 2025

### ERSSTv6 URL and file path
url = 'https://www.ncei.noaa.gov/data/sea-surface-temperature-extended-reconstructed/v6/access/'
output_path = os.path.join(data_dir, f'ERSSTv6_Global_{year_min}-{year_max}_SST_Monthly.nc')

print(f'\n---- Download ERSSTv6 Monthly Sea Surface Temperature (SST) ----\n')

### Loop through each years and months to download the ERSSTv6 data
datasets = []
for year in range(year_min, year_max + 1):
    for month in range(1, 13):
        filename = f'ersst.v6.{year}{month:02d}.nc'
        file_url = url + filename

        ### Download the monthly file into memory
        response = requests.get(file_url)
        response.raise_for_status()

        ### Save the downloaded file temporarily in the data directory
        temp_path = os.path.join(data_dir, filename)
        with open(temp_path, 'wb') as f:
            f.write(response.content)

        ### Append the downloaded dataset to the list
        with xr.open_dataset(temp_path) as ds:
            datasets.append(ds.load())

### Combine and save
ds = xr.concat(datasets, dim='time').sortby('time').squeeze('lev', drop=True)
ds = ds[['sst']].transpose('time', 'lat', 'lon').load()
ds.to_netcdf(output_path)
print(f'Saved {output_path}\n')

### Remove temporary files
for year in range(year_min, year_max + 1):
    for month in range(1, 13):

        ### Construct the ERSSTv6 filename
        filename = f'ersst.v6.{year}{month:02d}.nc'
        temp_path = os.path.join(data_dir, filename)

        ### Remove the file if it was successfully downloaded
        if os.path.exists(temp_path):
            os.remove(temp_path)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')