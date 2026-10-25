"""
Retrieve the monthly Relative Oceanic Niño Index (RONI), the three-month running mean of the relative Niño 3.4
index (ERSSTv5 Niño 3.4 minus tropical mean SST anomalies), from the NOAA Climate Prediction Center (CPC)

Notes
-----
    Data : https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt
    Author : Julian B Arnheim
"""

### Import modules
import os
import requests
import time

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data', 'raw')

### Variable
var = 'RONI'

### Years
year_min, year_max = 1950, 2026

### CPC RONI URL and file path
url = 'https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt'
output_path = os.path.join(data_dir, f'CPC_Nino34_{year_min}-{year_max}_{var}_Monthly_raw.txt')

print(f'\n---- Download CPC Monthly Relative Oceanic Niño Index (RONI) ----\n')

### Download
print(f'Downloading CPC monthly {var} ({year_min}–{year_max}) ...')
response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=60)
response.raise_for_status()

### Save untouched text file
with open(output_path, 'wb') as f:
    f.write(response.content)
print(f'Saved {output_path}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')