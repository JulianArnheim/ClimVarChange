"""
Retrieve the monthly Oceanic Niño Index (ONI), the three-month running mean of National Oceanic and Atmospheric Administration (NOAA)
Extended Reconstructed Sea Surface Temperature (ERSSTv6) anomalies in the Niño 3.4 region (5°N–5°S, 120°–170°W), from the NOAA Physical Sciences Laboratory (PSL)

Notes
-----
    Data : https://psl.noaa.gov/data/correlation/oni.data
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
var = 'ONI'

### Years
year_min, year_max = 1950, 2026

### PSL ONI URL and file path
url = 'https://psl.noaa.gov/data/correlation/oni.data'
output_path = os.path.join(data_dir, f'PSL_Nino34_{year_min}-{year_max}_{var}_Monthly_raw.txt')

print(f'\n---- Download PSL Monthly Oceanic Niño Index (ONI) ----\n')

### Download
print(f'Downloading PSL monthly {var} ({year_min}–{year_max}) ...')
response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=60)
response.raise_for_status()

### Save untouched text file
with open(output_path, 'wb') as f:
    f.write(response.content)
print(f'Saved {output_path}\n')

### End time
print(f'\n---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')