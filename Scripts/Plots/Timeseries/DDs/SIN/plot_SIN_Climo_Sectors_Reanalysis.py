"""
Plot Sinuosity Index (SIN) climatology from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib.ticker import FixedFormatter
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
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/DDs/SIN/')

### Years
year_min, year_max = 1940, 2025

### Epochs
epochs = [(year_min, year_max), (1981, 2010)]

print(f'\n---- Sinuosity Index (SIN) Climatology ----\n')

### Load SIN data
SIN_datasets = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_SIN_Sectors_{start_year}-{end_year}_Daily.nc')
    if os.path.exists(filename):
        SIN_datasets.append(xr.open_dataset(filename))
        print(f'Loaded: {start_year}–{end_year}')
SINs = xr.concat(SIN_datasets, dim='time')
print(f'Concatenated SINs\n')

### Loop through epochs
for start_yr, end_yr in epochs:

    ### Time specification
    SINs_time = SINs.sel(time=slice(f'{start_yr}-01-01', f'{end_yr}-12-31'))

    ### Time axis specifications
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
    month_positions = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349, 349 + 30]
    month_positions = [pos - 14 for pos in month_positions]

    ### Sectors
    sector_options = ['NH (180°W–180°E)', 'AM (140°W–50°W)', 'AT (50°W–40°E)', 'AS (40°E–130°E)', 'PA (130°E–140°W)']

    ### Colors for each sector
    colors = {'NH (180°W–180°E)': 'k', 'AM (140°W–50°W)': 'navy', 'AT (50°W–40°E)': 'forestgreen', 'AS (40°E–130°E)': 'firebrick', 'PA (130°E–140°W)': 'darkorange'}

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
    for key in ['top', 'right']:
        ax.spines[key].set_color('none')
    for key in ['bottom', 'left']:
        ax.spines[key].set_linewidth(2)
    ax.tick_params(axis='both', length=5.5, width=2, which='major', colors='dimgray')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('k')

    ### Horizontal lines
    for i in np.arange(1, 3, .1):
        plt.axhline(i, color='gray', linestyle='solid' if np.isclose(i, 2) else 'dotted', linewidth=.5, zorder=1)

    ### Process SIN climatology
    SINs_daily = SINs_time.sortby('time')
    climo_SINs_daily_year = SINs_daily.groupby('time.dayofyear').mean(dim='time').sel(dayofyear=slice(1, 365))

    ### Plot
    for sector in sector_options:
        sector_key = f'SIN_{sector.split(" ")[0]}'
        days = climo_SINs_daily_year['dayofyear']
        timeseries = climo_SINs_daily_year[sector_key]
        if sector.startswith('NH'):
            plt.plot(days, timeseries, color=colors[sector], label=sector, linewidth=3, zorder=3)
        else:
            plt.plot(days, timeseries, color=colors[sector], label=sector, linewidth=3, zorder=2)

    ### Title
    plt.title(r'\textbf{Sinuosity Index (SIN)}', color='k', fontsize=25, y=1.01)

    ### Ticks
    sinuosity_ticks = np.arange(0, 5, .1)
    sinuosity_ticks_labels = [f'{tick:.1f}' for tick in sinuosity_ticks]
    plt.xticks(month_positions, month_labels, fontsize=12, color='k', rotation=0)
    plt.yticks(sinuosity_ticks, sinuosity_ticks, fontsize=12, color='k')
    plt.gca().yaxis.set_major_formatter(FixedFormatter(sinuosity_ticks_labels))

    ### Domain
    plt.axis([1, 365, 1.2, 2.1])

    ### Label
    plt.xlabel(r'\textbf{Month}', fontsize=16, color='k', labelpad=10)

    ### Annotation
    plt.text(.5, .935, f'{start_yr}–{end_yr} Climatology', color='k', fontsize=14, ha='center', transform=plt.gca().transAxes)

    ### Legend
    l = plt.legend(shadow=False, fontsize=13, loc='upper center', bbox_to_anchor=(.78, .97), frameon=False, labelspacing=.46)
    for text in l.get_texts():
        text.set_color('k')

    ### Save figure
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, f'Reanalysis_SIN_{start_yr}–{end_yr}_Climo_Sectors.png')
    plt.savefig(fig_path, dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')