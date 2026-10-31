"""
Preprocess the Pan-Arctic Ice Modeling and Assimilation System (PIOMAS) assimilative ice-ocean model
monthly Sea Ice Thickness (SIT) from the University of Washington Polar Science Center

Notes
-----
    Data : https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/model_grid
    Author : Julian B Arnheim
"""

### Import modules
from scipy.interpolate import griddata
import numpy as np
import os
import time
import xarray as xr

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'Data'))
grid_file = os.path.join(data_dir, 'raw', 'grid.dat.txt')
output_file = os.path.join(data_dir, f'PIOMAS_Arctic_1979-2025_SIT_Monthly.nc')

### Years
year_min, year_max = 1979, 2025

### PIOMAS grid size
nlat, nlon = 120, 360

print(f'\n---- Preprocess PIOMAS SIT ----\n')

### Load PIOMAS curvilinear grid
if os.path.exists(grid_file):
    with open(grid_file, 'r') as f:
        grid_numbers = np.fromiter((float(x) for line in f for x in line.split()), dtype=np.float32)
    expected_points = nlat * nlon
    if grid_numbers.size != 2 * expected_points:
        raise ValueError(f'Unexpected number of values in grid file: {grid_numbers.size}, expected {2*expected_points}')
    lons = grid_numbers[:expected_points].reshape(nlat, nlon)
    lats = grid_numbers[expected_points:].reshape(nlat, nlon)
    print(f'Loaded grid: lats {lats.shape}, lons {lons.shape}\n')
else:
    raise FileNotFoundError(f'Grid file not found: {grid_file}')

### Target .25° x .25° grid north of 30°N, matching the PIOMAS SIC grid
target_lats, target_lons = np.arange(90, 29.75, -.25).astype(np.float32), np.arange(0, 360, .25).astype(np.float32)
target_xy = (90 - target_lats)[:, None] * np.exp(1j * np.deg2rad(target_lons))[None, :]  # azimuthal equidistant x/y about the pole avoids seam and pole artefacts

### Loop through each year
all_data = []
for year in range(year_min, year_max + 1):
    filename = os.path.join(data_dir, 'raw', f'heff.H{year}')
    if os.path.exists(filename):
        data = np.fromfile(filename, dtype=np.float32)
        data = data.reshape((12, nlat, nlon))
        data[data <= .15] = np.nan  # mask open water and grid-cell mean (effective) thickness of .15 m or less
        all_data.append(data)
        print(f'Loaded: {filename} -> shape {data.shape}')
    else:
        print(f'File not found: {filename}')
if not all_data:
    raise ValueError('No PIOMAS data loaded.')

### Concatenate all years along time axis
all_data = np.concatenate(all_data, axis=0)
ntime = all_data.shape[0]
print(f'\nConcatenated all years -> shape: {all_data.shape}\n')

### Regrid the valid PIOMAS domain to .25° grid
valid_idx = np.where(lats >= 45.0)  # only consider PIOMAS domain ~45–90°N for interpolation
regridded_data = np.full((ntime, len(target_lats), len(target_lons)), np.nan, dtype=np.float32)
flat_points = np.column_stack(((90 - lats[valid_idx]) * np.cos(np.deg2rad(lons[valid_idx])), (90 - lats[valid_idx]) * np.sin(np.deg2rad(lons[valid_idx]))))
for t in range(ntime):
    flat_values = all_data[t][valid_idx]
    regridded_data[t] = griddata(flat_points, flat_values, (target_xy.real, target_xy.imag), method='linear')
    if t % 50 == 0:
        print(f'Regridded month {t+1}/{ntime}')

### Generate monthly datetime64 coordinates
months = np.arange(np.datetime64(f'{year_min}-01'), np.datetime64(f'{year_max + 1}-01'), np.timedelta64(1, 'M'))
if months.size != ntime:
    raise ValueError(f'Mismatch between time steps ({ntime}) and generated months ({months.size})')

### Create xarray Dataset
ds = xr.Dataset({'SIT': (('time', 'latitude', 'longitude'), regridded_data)},
                coords={'time': months, 'latitude': target_lats, 'longitude': target_lons})
ds['SIT'].attrs = {'long_name': 'Sea ice thickness (grid-cell mean, values of .15 m or less masked)', 'units': 'm'}

### Save to NetCDF
ds.to_netcdf(output_file)
print(f'Saved {output_file}\n{ds}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')