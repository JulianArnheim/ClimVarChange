"""
Plot timeseries of Blocking Index (BLO) blocked point latitudes from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import matplotlib.pyplot as plt
import numpy as np
import os
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

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/DDs/BLO/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Blocking Index (BLO) Latitude Timeseries ----\n')

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Load BLO data
BLO_datasets = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_BLO_1d_{start_year}-{end_year}_Daily.nc')
    if os.path.exists(filename):
        BLO_datasets.append(xr.open_dataset(filename))
        print(f'Loaded: {start_year}–{end_year}')
BLOs = xr.concat(BLO_datasets, dim='time')
print(f'Concatenated BLOs\n')

### 5-day persistence criteria: 1d -> 5d
a = 5
window_end = (BLOs['BLO'].sortby('time').rolling(time=a, min_periods=a).sum() >= a).astype(np.int8)  # 1 on the last day of each fully blocked a-day window
BLO_binary_5d_latlon = (window_end.rolling(time=a, min_periods=1).max().shift(time=1 - a, fill_value=0) >= 1).astype(np.int8)  # extend the flag back over the window

### Loop through each season
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### First valid year
    first_year = year_min + 1 if text_months == 'DJF' else year_min

    ### Figure
    print(f'{text_months}')

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
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both', length=5.5, width=2, which='major')

    ### Horizontal lines
    for i in np.arange(30, 95, 5):
        plt.axhline(i, color='gray', linestyle='dotted', linewidth=.5)

    ### Sectors
    sectors = {'NH (180°W–180°E)': {'mask': lambda lon: (lon >= -180) & (lon < 180), 'color': 'k'},
               'AM (140°W–50°W)': {'mask': lambda lon: (lon >= -140) & (lon <= -50), 'color': 'navy'},
               'AT (50°W–40°E)': {'mask': lambda lon: ((lon >= -50) & (lon < 0)) | ((lon >= 0) & (lon <= 40)), 'color': 'forestgreen'},
               'AS (40°E–130°E)': {'mask': lambda lon: (lon >= 40) & (lon <= 130), 'color': 'firebrick'},
               'PA (130°E–140°W)': {'mask': lambda lon: ((lon >= 130) & (lon <= 180)) | ((lon >= -180) & (lon <= -140)), 'color': 'darkorange'}}

    for i, (sector_name, info) in enumerate(sectors.items()):

        ### Process BLO blocked latitudes
        BLO_condition = (BLO_binary_5d_latlon == 1) & info['mask'](BLO_binary_5d_latlon['longitude'])
        sector_data = BLO_binary_5d_latlon['latitude'].where(BLO_condition).mean(dim=['latitude', 'longitude'])
        sector_data = sector_data.sel(time=sector_data['time.month'].isin(mons))

        ### Process BLO means
        if text_months == 'DJF':
            month, year = sector_data.time.dt.month, sector_data.time.dt.year
            winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
            sector_data = sector_data.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
            sector_data = sector_data.groupby('winter_year').mean(dim='time').sel(winter_year=slice(first_year, year_max)).rename({'winter_year': 'year'})
        else:
            sector_data = sector_data.groupby('time.year').mean(dim='time').sel(year=slice(first_year, year_max))

        ### Plot
        plt.plot(sector_data['year'], sector_data, label=sector_name, color=info['color'], linewidth=3 if i == 0 else 2, zorder=5 if i == 0 else 3)

    ### Ticks
    plt.xticks(np.arange(1940, 2026, 5), np.arange(1940, 2026, 5), fontsize=11, color='k')
    plt.yticks(np.arange(30, 95, 5), [f'{deg}°' for deg in np.arange(30, 95, 5)], fontsize=12, color='k')

    ### Domain
    plt.axis([1940, 2025, 35, 80])

    ### Title
    plt.title(rf'\textbf{{Mean Latitude of {"" if text_months == "ANNUAL" else f"{text_months} "}Blocked Points}}', color='k', fontsize=22)

    ### Labels and legend
    plt.xlabel(r'\textbf{Year}', fontsize=14, color='k', labelpad=10); plt.ylabel(r'\textbf{Latitude [°N]}', fontsize=14, color='k', labelpad=10)
    l = plt.legend(shadow=False, fontsize=10, loc='upper center', bbox_to_anchor=(.5, .988), ncol=6, frameon=False, handlelength=1.2, columnspacing=1.1)
    for text in l.get_texts():
        text.set_color('k')

    ### Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_NH_BLO_{text_months}_Latitude_{first_year}–{year_max}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')