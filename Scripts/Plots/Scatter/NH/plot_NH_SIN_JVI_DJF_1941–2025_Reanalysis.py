"""
Plot scatterplot between monthly-mean mid-latitude circulation diagnostics from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt, matplotlib.ticker as ticker
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
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Scatter/NH/')

### Years
year_min, year_max = 1940, 2025
first_year = year_min + 1

print(f'\n---- Sinuosity Index (SIN) & Jet Variability Index (JVI) ----\n')

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='k')
plt.rc('xtick', color='dimgray', labelcolor='k')
plt.rc('ytick', color='dimgray', labelcolor='k')

### Create figure
fig, ax = plt.subplots(figsize=(5.5, 4.5))

### Adjust axes
adjust_spines(ax, ['left', 'bottom'])
for key in ['top', 'right']:
    ax.spines[key].set_color('none')
for key in ['bottom', 'left']:
    ax.spines[key].set_linewidth(2)
ax.tick_params(axis='both', length=5, width=2, which='major', colors='dimgray')
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color('k')
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))

### Grid lines
for i in np.arange(0, 5.5, .1):
    plt.axvline(i, color='dimgray', linestyle='solid' if np.isclose(i * 2, round(i * 2)) else 'dotted', linewidth=.5, label='_nolegend_', zorder=3)
for j in np.arange(10, 50, 1):
    plt.axhline(j, color='dimgray', linestyle='solid' if j % 5 == 0 else 'dotted', linewidth=.5, label='_nolegend_', zorder=3)

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

### Load JVI data
JVI_datasets = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_JVI_Sectors_{start_year}-{end_year}_Daily.nc')
    if os.path.exists(filename):
        JVI_datasets.append(xr.open_dataset(filename))
        print(f'Loaded: {start_year}–{end_year}')
JVIs = xr.concat(JVI_datasets, dim='time')
print(f'Concatenated JVIs\n')

### Process winter data
SIN_data = SINs.sel(time=slice(f'{year_min}-01-01', f'{year_max}-12-31')).where(SINs.time.dt.month.isin([12, 1, 2]), drop=True)
JVI_data = JVIs.sel(time=slice(f'{year_min}-01-01', f'{year_max}-12-31')).where(JVIs.time.dt.month.isin([12, 1, 2]), drop=True)
month, year = SIN_data.time.dt.month, SIN_data.time.dt.year
winter_year = xr.where(month == 12, year + 1, year)
SIN_data = SIN_data.assign_coords(winter_year=('time', winter_year.data))
JVI_data = JVI_data.assign_coords(winter_year=('time', winter_year.data))
valid_years = [yr for yr, grp in SIN_data.groupby('winter_year') if len(np.unique(grp.time.dt.month)) == 3]
SIN_data = SIN_data.sel(time=SIN_data.winter_year.isin(valid_years))
JVI_data = JVI_data.sel(time=JVI_data.winter_year.isin(valid_years))

### Statistics
print(f'DJF winters: {valid_years[0]}–{valid_years[-1]} ({len(valid_years)} seasons)')
max_SIN, min_SIN = round(SIN_data['SIN_NH'].max().item(), 5), round(SIN_data['SIN_NH'].min().item(), 5)
max_JVI, min_JVI = round(JVI_data['JVI_NH'].max().item(), 5), round(JVI_data['JVI_NH'].min().item(), 5)
print(f'SIN (min, max): ({min_SIN}, {max_SIN})')
print(f'JVI (min, max): ({min_JVI}, {max_JVI})\n')

### Colorbar
cmap = colormaps.get_cmap('viridis')
norm = BoundaryNorm(np.arange(0, 301, 25), ncolors=cmap.N, extend='max')

### Scatter
hist = plt.hist2d(SIN_data['SIN_NH'].values.flatten(), JVI_data['JVI_NH'].values.flatten(), bins=[np.arange(1, 5.1, .1), np.arange(10, 40.1, 1)], cmap=cmap, cmin=1, norm=norm)
ticks = np.arange(0, 301, 100)
cbar = plt.colorbar(ticks=ticks, extend='max')
cbar.set_label('Number of Months', fontsize=11, fontname='AvantGarde', rotation=270, labelpad=18)
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_yticklabels([f'{v:g}' for v in ticks])
cbar.minorticks_off()
cbar.ax.set_position([.83, .02, .025, .98])

### Title
plt.title(r'\textbf{SIN vs JVI}', color='k', fontsize=16, y=1)

### Ticks
plt.xticks(np.arange(1, 5.1, .5), fontsize=8, fontname='Avant Garde')
plt.yticks(np.arange(10, 41, 5), fontsize=8, fontname='Avant Garde')
plt.tick_params(axis='both', labelcolor='k', color='dimgray')

### Labels and annotations
plt.xlabel(r'\textbf{SIN}', fontsize=12, color='k', labelpad=8); plt.ylabel(r'\textbf{JVI}', fontsize=12, color='k', labelpad=8)
plt.text(1, 1.01, f'{first_year}–{year_max}', color='k', fontsize=8, ha='right', transform=plt.gca().transAxes)
plt.text(1.01, .07, r'\textbf{DJF}', transform=ax.transAxes, fontsize=52, color='gray', fontweight='bold', ha='right', va='center', alpha=1, zorder=10)

### Domain
plt.axis([1, 2.5, 25, 40])

### Save figure
ax.set_position([.15, .15, .65, .72])
plt.savefig(os.path.join(fig_dir, f'Reanalysis_NH_SIN_JVI_DJF_{first_year}–{year_max}.png'), dpi=1000)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')