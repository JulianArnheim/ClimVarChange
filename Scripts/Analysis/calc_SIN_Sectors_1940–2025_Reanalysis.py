"""
Calculate Sinuosity Index (SIN) from Barnes (2013), Cattiaux et al. (2016),
Peings et al. (2017, 2018), Arnheim et al. (2025) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from skimage import measure
import numpy as np
import os
import time
import xarray as xr

def cosine_weighted_spatial_mean(data, coord=None):
    coord = coord or ('lat' if 'lat' in data.coords else 'latitude' if 'latitude' in data.coords else None)
    if not coord:
        raise ValueError("No 'lat' or 'latitude' coordinate found in the data array.")
    if 'lat' in data.coords and 'latitude' in data.coords:
        raise ValueError("Both 'lat' and 'latitude' coordinates exist.")
    weights = np.cos(np.deg2rad(data[coord]))
    return (data * weights).sum(dim=coord) / (data.notnull() * weights).sum(dim=coord)

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(base_dir, 'Data')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Sinuosity Index (SIN) ----\n')

for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)

    output_file_decade = os.path.join(data_dir, f'ERA5_SIN_Sectors_{start_year}-{end_year}_Daily.nc')
    if os.path.exists(output_file_decade):
        print(f'  Skipping {start_year}–{end_year}, file exists.')
        continue

    filename = os.path.join(data_dir, f'ERA5_NH_{start_year}-{end_year}_Z500_Daily.nc')
    if not os.path.exists(filename):
        print(f'  Data file missing: {filename}, skipping.')
        continue

    print(f'\n--- Processing Decade {start_year}–{end_year} ---\n')

    ### Load decade data
    dataz = xr.open_dataset(filename, chunks={'time': 10})['z'].sortby('time')

    ### Define longitude sectors using boolean masks
    longitudes = dataz['longitude']
    sectors = {'NH': (longitudes >= -180) & (longitudes < 180),
               'AM': (longitudes >= -140) & (longitudes <= -50),
               'AT': (longitudes >= -50) & (longitudes <= 40),
               'AS': (longitudes >= 40) & (longitudes <= 130),
               'PA': ((longitudes >= 130) | (longitudes <= -140))}

    ### Compute SIN for each sector and decade
    sinuosity_data = {}
    for key, mask in sectors.items():
        print(f'  Sector: {key}')

        ### Initializations
        values = np.empty(dataz.time.size)
        prev_year = None

        ### Loop through each day in the decade
        for t in range(dataz.time.size):

            ### Print progress whenever a new year
            curr_year = dataz.time.dt.year.values[t]
            if curr_year != prev_year:
                print(f'    Year: {curr_year}')
                prev_year = curr_year

            ### Sector and latitudinal band specification
            if key == 'PA':
                z = dataz.isel(time=t).sel(longitude=longitudes[mask])
                z = z.assign_coords(longitude=xr.where(z.longitude < 0, z.longitude + 360, z.longitude)).sortby('longitude')
            else:
                z = dataz.isel(time=t).sel(longitude=longitudes[mask])

            ### Compute spatial mean
            lat_name = 'lat' if 'lat' in dataz.dims else 'latitude'
            z_mean = cosine_weighted_spatial_mean(z.sel({lat_name: slice(70, 30)}).mean(dim='longitude'))  # mean over 30–70°N

            ### Compute total contour length on the sphere for the contour at the spatial mean
            contours = measure.find_contours(z.values, z_mean)
            lats, lons = z[lat_name].values, z.longitude.values
            contour_length_total = 0
            for c in contours:

                ### Contour vertices from grid indices to latitude and longitude [rad]
                phi = np.deg2rad(np.interp(c[:, 0], np.arange(lats.size), lats))
                lam = np.deg2rad(np.interp(c[:, 1], np.arange(lons.size), lons))

                ### Great-circle length of each segment on the unit sphere (haversine)
                segment_lengths = 2 * np.arcsin(np.sqrt(np.sin(np.diff(phi) / 2)**2 + np.cos(phi[:-1]) * np.cos(phi[1:]) * np.sin(np.diff(lam) / 2)**2))

                ### Sum segment lengths
                contour_length_total += np.sum(segment_lengths)

            ### Normalize by the length of the 50°N circle across the sector (Cattiaux et al. 2016)
            values[t] = contour_length_total / (np.deg2rad(lons.size * abs(lons[1] - lons[0])) * np.cos(np.deg2rad(50)))

        sinuosity_data[f'SIN_{key}'] = (['time'], values)

    ### Save SIN decade dataset
    ds_out = xr.Dataset(sinuosity_data, coords={'time': dataz.time.values})
    ds_out.to_netcdf(output_file_decade)
    print(f'  Saved {output_file_decade}, the SIN data\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')