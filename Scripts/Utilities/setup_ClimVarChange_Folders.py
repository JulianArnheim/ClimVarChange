"""
Create the Data/ and Figures/ folder layout listed in the README
so every download, preprocess, analysis, and plot script finds its output folder

Notes
-----
    Author : Julian B Arnheim
"""

### Import modules
import itertools
import os
import time

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

### Folder layout, with each list standing for a brace group in the README
regions = ['Global', 'NH']

### Top-level profile folders, with each list standing for a brace group in the README
tops = ['10 top', '100 top']

### Common pressure (height) levels
levs = [10, 100, 250, 500, 700, 850]

### Common map and profile categories
maps = ['Diff', 'Historical', 'Historical Std', 'Slope']

### Define the complete directory structure used throughout the repository
layout = [('Data', 'raw'),
          ('Figures', ['Bars', 'Table']),
          ('Figures', 'Histograms', ['JLI', 'SIN']),
          ('Figures', 'Profile', ['Amplification', 'Historical Std'], regions, tops),
          ('Figures', 'Profile', 'Diffs', regions, tops, ['T', 'TU', 'TV']),
          ('Figures', 'Profile', 'Historical', regions, tops, ['T', 'TPsi', 'TU', 'TV']),
          ('Figures', 'Profile', 'Slopes', regions, tops, ['T', 'TU', 'TV'], ['ANNUAL', 'DJF', 'JJA']),
          ('Figures', 'Profile', 'Slopes', 'Series', 'no_stippling'),
          ('Figures', 'Scatter', ['AT', 'NH']),
          ('Figures', 'Spatial', 'Arctic', ['ICEFRAC', 'SIT'], maps),
          ('Figures', 'Spatial', 'Arctic', 'T2M', 'Local AA', ['Latitudes', 'Periods', 'Sweeney2023']),
          ('Figures', 'Spatial', 'Global', [os.path.join('OMEGA', 'OMEGA200'), os.path.join('OMEGA', 'OMEGA500'), 'PRECT', 'SLP', 'SST', 'T2M'], maps),
          ('Figures', 'Spatial', 'Global', [f'UV{lev}' for lev in levs], 'Historical'),
          ('Figures', 'Spatial', 'NA', ['Baroclinicity', 'NAO', 'Regimes', 'Stormtrack']),
          ('Figures', 'Spatial', 'NA', 'SST', 'NAWH', ['ERA5', 'ERSSTv6', 'HadISST']),
          ('Figures', 'Spatial', 'NH', ['BLO', 'SIN']),
          ('Figures', 'Spatial', 'NH', 'T2M', maps),
          ('Figures', 'Spatial', 'CONUS', 'T2M', 'Slope', 'no_stippling'),
          ('Figures', 'Timeseries', 'CCIs', ['letters', 'no letters'], 'anom'),
          ('Figures', 'Timeseries', 'DDs', ['BLO', 'JVI', 'SIN', 'ZON']),
          ('Figures', 'Timeseries', ['ENSO', os.path.join('NAO', 'gifs'), 'Z500']),
          ('Figures', 'Timeseries', 'ICEFRAC', ['SIA', 'SIE', 'SIV']),
          ('Figures', 'Timeseries', ['PRECT', 'T2M'], ['Arctic', 'Global', 'NH']),
          ('Figures', 'Timeseries', 'SST', ['Arctic', 'Global', 'NAWH']),
          ('Figures', 'Zonal Mean', ['SST', 'T2M'])]

print(f'\n---- Create Data and Figures Folders ----\n')

### Create every folder in the layout, leaving existing folders and their contents untouched
for parts in layout:
    for combo in itertools.product(*[part if isinstance(part, list) else [part] for part in parts]):
        folder = os.path.join(base_dir, *combo)
        existed = os.path.isdir(folder)
        os.makedirs(folder, exist_ok=True)
        print(f'{"Exists" if existed else "Created"}: {os.path.relpath(folder, base_dir)}')

### End time
print(f'\n---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')