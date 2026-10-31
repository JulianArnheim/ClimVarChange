"""
Plot mean North Atlantic Warming Hole (NAWH) region (Park & Yeh 2024) Sea Surface Temperature (SST)
from ERA5 reanalysis and the HadISST and ERSSTv6 analyses

Notes
-----
    Data : (ERA5) https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Data : (Met Office) https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html
    Data : (NOAA) https://www.ncei.noaa.gov/products/extended-reconstructed-sst
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib.patches import Polygon
from shapely.geometry import LineString, Point
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.pyplot as plt, matplotlib.ticker as ticker
import numpy as np
import os
import re
import time
import xarray as xr

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks_position('bottom')

def crop_region_mean(ds, lat_min, lat_max, lon_min, lon_max):
    lat_coord = 'lat' if 'lat' in ds.coords else 'latitude'
    lon_coord = 'lon' if 'lon' in ds.coords else 'longitude'
    ds = ds.where((ds[lat_coord] >= lat_min) & (ds[lat_coord] <= lat_max) & (ds[lon_coord] >= lon_min) & (ds[lon_coord] <= lon_max), drop=True)
    weights = np.cos(np.deg2rad(ds[lat_coord])).broadcast_like(ds['sst'])
    return (ds['sst'] * weights).sum((lat_coord, lon_coord), skipna=True) / weights.where(ds['sst'].notnull()).sum((lat_coord, lon_coord))

def custom_formatter(x, _):
    if abs(x) < .01:
        return '0'
    if x == int(x):
        return f'{x:+.0f}'
    return f'{x:+.1f}'.replace('+0.', '+.').replace('-0.', '-.')

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

def longitude_360(ds, lon_name='lon'):
    if ds[lon_name].min() < 0:
        ds = ds.assign_coords({lon_name: ds[lon_name] % 360}).sortby(lon_name)
    return ds

def reestablish_time(ds, filename, var_name='sst'):
    var_lookup = {v.lower(): v for v in ds.data_vars}
    if var_name.lower() not in var_lookup:
        raise ValueError(f"Variable '{var_name}' not found in dataset.")
    var_name_actual = var_lookup[var_name.lower()]
    ds = ds[[var_name_actual]]
    ds = ds.drop_dims([dim for dim in ds.dims if dim not in ds[var_name_actual].dims])
    base_filename = os.path.basename(filename)
    match = re.search(r'(\d{4})', base_filename)
    if not match:
        raise ValueError(f'No 4-digit year found in filename: {base_filename}')
    time_index = ds['time'].values.astype('datetime64[M]')  # keep the file's own dates, normalized to the first of each month
    if int(str(time_index[0])[:4]) != int(match.group(1)):
        print(f'Warning: {base_filename} starts in {str(time_index[0])[:4]}, not {match.group(1)}')
    return ds.assign_coords(time=time_index)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/SST/NAWH/')

### Years
year_min, year_max = 1940, 2025

### Epoch
epoch = (1981, 2010)

### Box definition (NAWH) as defined in Park & Yeh (2024) and Park et al. (2024): [45°-65°N, 50°-15°W]
lat_min, lat_max = 45, 65
lon_min, lon_max = 310, 345  # 0-360 format

print(f'\n---- North Atlantic Warming Hole (NAWH) ----\n')

### Load data
ERA5_filename = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_SST_Monthly.nc')
ERA5_datat = xr.open_dataset(ERA5_filename)
HadISST_filename = os.path.join(data_dir, f'HadISST_Global_1870-{year_max}_SST_Monthly.nc')
HadISST_datat = xr.open_dataset(HadISST_filename)
ERSSTv6_filename = os.path.join(data_dir, f'ERSSTv6_Global_1850-{year_max}_SST_Monthly.nc')
ERSSTv6_datat = xr.open_dataset(ERSSTv6_filename)

### Reestablish time coordinate
ERA5_datat = reestablish_time(ERA5_datat, ERA5_filename)
HadISST_datat = reestablish_time(HadISST_datat, HadISST_filename)
ERSSTv6_datat = reestablish_time(ERSSTv6_datat, ERSSTv6_filename)

### Ensure physical values
ERA5_datat['sst'] = ERA5_datat['sst'].where((ERA5_datat['sst'] > -10) & (ERA5_datat['sst'] < 50))
HadISST_datat['sst'] = HadISST_datat['sst'].where((HadISST_datat['sst'] > -10) & (HadISST_datat['sst'] < 50))
ERSSTv6_datat['sst'] = ERSSTv6_datat['sst'].where((ERSSTv6_datat['sst'] > -10) & (ERSSTv6_datat['sst'] < 50))

### Standardize longitude coordinates
ERA5_datat = longitude_360(ERA5_datat, lon_name='lon' if 'lon' in ERA5_datat.coords else 'longitude')
HadISST_datat = longitude_360(HadISST_datat, lon_name='lon' if 'lon' in HadISST_datat.coords else 'longitude')
ERSSTv6_datat = longitude_360(ERSSTv6_datat, lon_name='lon' if 'lon' in ERSSTv6_datat.coords else 'longitude')

### Crop
ERA5_datat = ERA5_datat.sel(time=slice(f'{year_min}-01-01', f'{year_max}-12-31'))[['sst']]
HadISST_datat = HadISST_datat.sel(time=slice(f'{year_min}-01-01', f'{year_max}-12-31'))[['sst']]
ERSSTv6_datat = ERSSTv6_datat.sel(time=slice(f'{year_min}-01-01', f'{year_max}-12-31'))[['sst']]

### Season
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each season
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### For title only, make ANNUAL blank
    season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '

    ### Figure
    print(f'{text_months}')

    ### Compute or load timeseries
    era5_path = os.path.join(data_dir, f'ERA5_NAWH_SST_{text_months}_{year_min}-{year_max}_Annual.nc')
    hadisst_path = os.path.join(data_dir, f'HadISST_NAWH_SST_{text_months}_{year_min}-{year_max}_Annual.nc')
    ersstv6_path = os.path.join(data_dir, f'ERSSTv6_NAWH_SST_{text_months}_{year_min}-{year_max}_Annual.nc')

    if os.path.exists(era5_path) and os.path.exists(hadisst_path) and os.path.exists(ersstv6_path):
        print(f'Timeseries files for {text_months} already exist. Loading...')
        ERA5_anom = xr.open_dataset(era5_path)['sst']
        HadISST_anom = xr.open_dataset(hadisst_path)['sst']
        ERSSTv6_anom = xr.open_dataset(ersstv6_path)['sst']

    else:
        ### Season specification
        ERA5_selected = ERA5_datat.sel(time=ERA5_datat['time.month'].isin(mons))
        HadISST_selected = HadISST_datat.sel(time=HadISST_datat['time.month'].isin(mons))
        ERSSTv6_selected = ERSSTv6_datat.sel(time=ERSSTv6_datat['time.month'].isin(mons))

        ### Annual mean
        if text_months == 'DJF':
            first_year = year_min + 1

            ### ERA5
            month, year = ERA5_selected.time.dt.month, ERA5_selected.time.dt.year
            winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
            ERA5_annual = ERA5_selected.assign_coords(winter_year=('time', winter_year.data))
            ERA5_annual = ERA5_annual.where(~np.isnan(winter_year), drop=True).groupby('winter_year').mean(dim='time')
            ERA5_annual = ERA5_annual.sel(winter_year=slice(first_year, year_max)).rename({'winter_year': 'year'})

            ### HadISST
            month, year = HadISST_selected.time.dt.month, HadISST_selected.time.dt.year
            winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
            HadISST_annual = HadISST_selected.assign_coords(winter_year=('time', winter_year.data))
            HadISST_annual = HadISST_annual.where(~np.isnan(winter_year), drop=True).groupby('winter_year').mean(dim='time')
            HadISST_annual = HadISST_annual.sel(winter_year=slice(first_year, year_max)).rename({'winter_year': 'year'})

            ### ERSSTv6
            month, year = ERSSTv6_selected.time.dt.month, ERSSTv6_selected.time.dt.year
            winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
            ERSSTv6_annual = ERSSTv6_selected.assign_coords(winter_year=('time', winter_year.data))
            ERSSTv6_annual = ERSSTv6_annual.where(~np.isnan(winter_year), drop=True).groupby('winter_year').mean(dim='time')
            ERSSTv6_annual = ERSSTv6_annual.sel(winter_year=slice(first_year, year_max)).rename({'winter_year': 'year'})

        else:
            ERA5_annual = ERA5_selected.groupby('time.year').mean(dim='time')
            HadISST_annual = HadISST_selected.groupby('time.year').mean(dim='time')
            ERSSTv6_annual = ERSSTv6_selected.groupby('time.year').mean(dim='time')

        ### Crop and compute spatial mean
        ERA5_box_mean = crop_region_mean(ERA5_annual, lat_min, lat_max, lon_min, lon_max)
        HadISST_box_mean = crop_region_mean(HadISST_annual, lat_min, lat_max, lon_min, lon_max)
        ERSSTv6_box_mean = crop_region_mean(ERSSTv6_annual, lat_min, lat_max, lon_min, lon_max)

        ### Compute climatology
        ERA5_baseline = ERA5_box_mean.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')
        HadISST_baseline = HadISST_box_mean.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')
        ERSSTv6_baseline = ERSSTv6_box_mean.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')

        ### Compute anomalies
        ERA5_anom, HadISST_anom = ERA5_box_mean - ERA5_baseline, HadISST_box_mean - HadISST_baseline
        ERSSTv6_anom = ERSSTv6_box_mean - ERSSTv6_baseline

        ### Change name
        ERA5_anom.name = 'sst'
        HadISST_anom.name = 'sst'
        ERSSTv6_anom.name = 'sst'

        ### Save anomaly files
        ERA5_anom.to_netcdf(era5_path)
        print(f'Saved {era5_path}, the ERA5 anomalies')
        HadISST_anom.to_netcdf(hadisst_path)
        print(f'Saved {hadisst_path}, the HadISST anomalies')
        ERSSTv6_anom.to_netcdf(ersstv6_path)
        print(f'Saved {ersstv6_path}, the ERSSTv6 anomalies')

    ### Figure design
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='w')
    plt.rc('axes', edgecolor='dimgray')
    plt.rc('xtick', color='dimgray')
    plt.rc('ytick', color='dimgray')
    plt.rc('axes', labelcolor='dimgray')
    plt.rc('axes', facecolor='w')

    ### Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    ### Adjust axes
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

    ### Horizontal lines
    for step, style, lw in [(1, 'solid', .5), (.25, 'dotted', .5)]:
        for y in np.arange(-10, 10, step):
            plt.axhline(y, color='gray', linestyle=style, linewidth=1 if y == 0 else lw)

    ### Plot
    ax.plot(ERA5_anom['year'], ERA5_anom, label=r'\textbf{ERA5} (.25°)', color='magenta', linewidth=2, linestyle='solid')
    ax.plot(HadISST_anom['year'], HadISST_anom, label=r'\textbf{HadISST} (1°)', color='magenta', linewidth=2, linestyle='dotted')
    ax.plot(ERSSTv6_anom['year'], ERSSTv6_anom, label=r'\textbf{ERSSTv6} (2°)', color='magenta', linewidth=2, linestyle='dashed')

    ### Ticks
    plt.xticks(np.arange(1940, 2040, 10), np.arange(1940, 2040, 10), color='k')
    plt.yticks(np.arange(-5, 6, .5), map(str, np.arange(-5, 6, .5)), color='k')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

    ### Domain
    plt.axis([1940, 2025, -1.5, 1.5])

    ### Title
    plt.title(rf'\textbf{{{season_prefix}NORTH ATLANTIC WARMING HOLE}}', color='k', fontsize=32, y=1.02)

    ### Labels and annotation
    plt.xlabel(r'\textbf{Year}', fontsize=15, color='k', labelpad=10)
    plt.ylabel(r'\textbf{Sea Surface Temperature Anomaly [°C]}', fontsize=15, color='k', labelpad=10)
    plt.text(1960, -1.46, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}}}', fontsize=12, rotation='horizontal', ha='right')

    ### Legend
    l = plt.legend(shadow=False, fontsize=12, loc='upper center', bbox_to_anchor=(.64, .988), fancybox=True, ncol=1, frameon=False, labelspacing=.15)
    for text in l.get_texts():
        text.set_color('k')

    plt.tight_layout()

    ### Inset region
    pos = [.35, .63, .25, .25]
    inset_ax = fig.add_axes(pos, projection=ccrs.NearsidePerspective(central_longitude=-37, central_latitude=63, satellite_height=50000000), zorder=3)
    inset_ax.set_global()
    inset_ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='gray', zorder=4)
    inset_ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='w', zorder=5)
    inset_ax.coastlines(resolution='50m', linewidth=.5, zorder=6)
    lats, lons = [-60, -30, 0, 30, 60], np.linspace(-180, 180, 361)
    for lat in lats:
        line = LineString([(lon, lat) for lon in lons])
        inset_ax.add_geometries([line], crs=ccrs.PlateCarree(), edgecolor='k', facecolor='none', linestyle='--', linewidth=.5, alpha=1, zorder=8)
    N_pole, S_pole = Point(0, 90), Point(0, -90)
    inset_ax.scatter([N_pole.x, S_pole.x], [N_pole.y, S_pole.y], transform=ccrs.PlateCarree(), color='k', s=1, alpha=1, zorder=9)

    ### Mark region
    interp_points = 1000
    plot_lon_min, plot_lon_max = lon_min - 360, lon_max - 360
    N_lons = np.linspace(plot_lon_min, plot_lon_max, interp_points)
    N_lats = np.full(interp_points, lat_max)
    S_lons = np.linspace(plot_lon_min, plot_lon_max, interp_points)
    S_lats, E_lons = np.full(interp_points, lat_min), np.full(interp_points, plot_lon_max)
    E_lats, W_lons = np.linspace(lat_min, lat_max, interp_points), np.full(interp_points, plot_lon_min)
    W_lats = np.linspace(lat_min, lat_max, interp_points)
    lon = np.concatenate([W_lons, N_lons, E_lons[::-1], S_lons[::-1]])
    lat = np.concatenate([W_lats, N_lats, E_lats[::-1], S_lats[::-1]])
    lines = np.column_stack([lon, lat])
    inset_ax.add_patch(Polygon(lines, facecolor='magenta', alpha=.2, edgecolor='none', transform=ccrs.PlateCarree(), zorder=7))
    for lons, lats in [(S_lons, S_lats), (N_lons, N_lats), (W_lons, W_lats), (E_lons, E_lats)]:
        inset_ax.plot(lons, lats, color='magenta', linewidth=1.5, transform=ccrs.PlateCarree(), zorder=8)

    ### Save figure
    fig_path = os.path.join(fig_dir, f'Reanalysis_NAWH_SST_{text_months}_{year_min}–{year_max}.png')
    plt.savefig(fig_path, dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')