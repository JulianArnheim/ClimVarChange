"""
Plot Blocking Index (BLO) climatology from ERA5 reanalysis

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
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/DDs/BLO/')

### Years
year_min, year_max = 1940, 2025

### Epochs
epochs = [(year_min, year_max), (1981, 2010)]

print(f'\n---- Blocking Index (BLO) Climatology ----\n')

### Load BLO data
BLO_datasets = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_BLO_5d_Sectors_{start_year}-{end_year}_Daily.nc')
    if os.path.exists(filename):
        BLO_datasets.append(xr.open_dataset(filename))
        print(f'Loaded: {start_year}–{end_year}')
BLOs = xr.concat(BLO_datasets, dim='time')
print(f'Concatenated BLOs\n')

### Loop through epochs
for start_yr, end_yr in epochs:

    ### Time specification
    BLOs_time = BLOs.sel(time=slice(f'{start_yr}-01-01', f'{end_yr}-12-31'))

    ### Time axis specifications
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
    month_positions = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349, 349 + 30]
    month_positions = [pos - 14 for pos in month_positions]

    ### Sectors
    sector_options = ['NH (180°W–180°E)', 'AM (140°W–50°W)', 'AT (50°W–40°E)', 'AS (40°E–130°E)', 'PA (130°E–140°W)']

    ### Colors for each sector
    colors = {'NH (180°W–180°E)': 'k',
              'AM (140°W–50°W)': 'navy',
              'AT (50°W–40°E)': 'forestgreen',
              'AS (40°E–130°E)': 'firebrick',
              'PA (130°E–140°W)': 'darkorange'}

    ### Figure design
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='w')
    plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='dimgray')
    plt.rc('xtick', color='dimgray')
    plt.rc('ytick', color='dimgray')

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
    for step, style in [(1, 'solid'), (.1, 'dotted')]:
        for y in np.arange(.1, 1.1, step):
            ax.axhline(y, color='gray', linestyle=style, linewidth=.5, zorder=1)

    ### Process BLO climatology
    climo_BLOs_daily_year = BLOs_time.sortby('time').groupby('time.dayofyear').mean(dim='time')
    climo_BLOs_daily_year = climo_BLOs_daily_year.sel(dayofyear=slice(1, 365))

    ### Plot
    for sector in sector_options:
        days = climo_BLOs_daily_year['dayofyear']
        timeseries = climo_BLOs_daily_year[f"BLO_{sector.split(' ')[0]}"]
        if sector.startswith('NH'):
            plt.plot(days, timeseries, color=colors[sector], label=sector, linewidth=3, zorder=3)
        else:
            plt.plot(days, timeseries, color=colors[sector], label=sector, linewidth=3, zorder=2)

    ### Title
    plt.title(r'\textbf{Blocking Index (BLO)}', color='k', fontsize=25, y=1.01)

    ### Ticks
    prob_ticks = np.arange(0, 1.1, .1)
    prob_ticks_labels = ['0' if t == 0 else (f'.{int(t*10)}' if t < 1 else '1') for t in prob_ticks]
    plt.xticks(month_positions, month_labels, fontsize=12, color='k', rotation=0)
    plt.yticks(prob_ticks, prob_ticks, fontsize=12, color='k')
    plt.gca().yaxis.set_major_formatter(FixedFormatter(prob_ticks_labels))

    ### Domain
    plt.axis([1, 365, .1, 1])

    ### Labels and annotation
    plt.xlabel(r'\textbf{Month}', fontsize=16, color='k', labelpad=10)
    plt.ylabel(r'\textbf{Probability}', fontsize=16, color='k', labelpad=10)
    plt.text(.55, .15, f'{start_yr}–{end_yr} Climatology', color='k', fontsize=14, ha='center', transform=plt.gca().transAxes)

    ### Legend
    l = plt.legend(shadow=False, fontsize=11, loc='upper center', bbox_to_anchor=(.59, .625), frameon=False, labelspacing=.68)
    for text in l.get_texts():
        text.set_color('k')

    ### Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_BLO_{start_yr}–{end_yr}_Climo_Sectors.png'), dpi=1000)
    plt.close(fig)

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')