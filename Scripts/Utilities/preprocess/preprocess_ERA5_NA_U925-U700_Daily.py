"""
Preprocess the European Centre for Medium-Range Weather Forecasts (ECMWF) Reanalysis version 5 (ERA5)
daily 925-700 hPa Zonal Wind (U) in the North Atlantic region [60-0°W; 15-75°N] dataset

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import os
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
year_min, year_max = 1940, 2025

### Pressure levels
pressure_levels = [str(p) for p in [925, 900, 875, 850, 825, 800, 775, 750, 700]]

print(f'\n---- Preprocess ERA5 ----\n')

### Loop through pressure levels individually
for lev in pressure_levels:

    ### Loop through decade/partial-decade files and preprocess individually
    for start_year in range(year_min, 2030, 10):
        end_year = min(start_year + 9, year_max)

        ### Files
        input_file = os.path.join(data_dir, 'raw', f'ERA5_NA_{start_year}-{end_year}_U{lev}_Daily_raw.nc')
        output_file =  os.path.join(data_dir, f'ERA5_NA_{start_year}-{end_year}_U{lev}_Daily.nc')
        if os.path.exists(input_file):
            ds = xr.open_dataset(input_file)
            print(f'Loaded: {start_year}–{end_year}')

            ### Rename valid_time to time
            ds = ds.rename({'valid_time': 'time'}) if 'valid_time' in ds.coords else ds

            ### Drop unwanted coordinates
            ds = ds.drop_vars([c for c in ['number', 'expver'] if c in ds.coords])

            ### Drop single-level pressure_level coordinate if present
            if 'pressure_level' in ds.coords:
                if ds.coords['pressure_level'].size == 1:
                    ds = ds.drop_vars('pressure_level')
                    ds = ds.squeeze('pressure_level', drop=True)

            ### Strip hour/minute/second from time
            day_index = ds.indexes['time'].normalize().values.astype('datetime64[D]')
            ds = ds.assign_coords(time=day_index)

            ### Convert coordinates that are float64 to float32
            for coord in ds.coords:
                if ds[coord].dtype == 'float64':
                    ds[coord] = ds[coord].astype('float32')

            ### Save
            ds.to_netcdf(output_file)
            print(f'Saved {output_file}')
            print(f'{ds}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')