"""
Retrieve the Pan-Arctic Ice Modeling and Assimilation System (PIOMAS) assimilative ice-ocean model
monthly Sea Ice Volume (SIV) from the University of Washington Polar Science Center

Notes
-----
    Data : https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/
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
var = 'SIV'

### Years
year_min, year_max = 1979, 2025

### PIOMAS monthly ice volume URL and file path
url = 'https://psc.apl.uw.edu/wordpress/wp-content/uploads/schweiger/ice_volume/PIOMAS.monthly.Current.v2.1.csv'
output_path = os.path.join(data_dir, f'PIOMAS_Arctic_{year_min}-{year_max}_{var}_Monthly_raw.csv')

print(f'\n---- Download PIOMAS Monthly Sea Ice Volume (SIV) ----\n')

### Download
print(f'Downloading PIOMAS monthly {var} ({year_min}–{year_max}) ...')
response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=60)
response.raise_for_status()

### Save untouched CSV
with open(output_path, 'wb') as f:
    f.write(response.content)
print(f'Saved {output_path}\n')

### End time
print(f'\n---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')