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
fig_dir = os.path.join(base_dir, 'Figures/Scatter/AT/')

### Years
year_min, year_max = 1940, 2025

### Epoch
epoch_start, epoch_end = 1981, 2010

print(f'\n---- Zonal Index (ZON) & Jet Variability Index (JVI) ----\n')

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
for i in np.arange(0, 1000, 25):
    plt.axvline(i, color='dimgray', linestyle='solid' if i % 100 == 0 else 'dotted', linewidth=.5, label='_nolegend_', zorder=3)
for j in np.arange(0, 50, 1):
    plt.axhline(j, color='dimgray', linestyle='solid' if j % 5 == 0 else 'dotted', linewidth=.5, label='_nolegend_', zorder=3)

### Load ZON data
ZON_datasets = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_ZON_Sectors_{start_year}-{end_year}_Daily.nc')
    if os.path.exists(filename):
        ZON_datasets.append(xr.open_dataset(filename))
        print(f'Loaded: {start_year}–{end_year}')
ZONs = xr.concat(ZON_datasets, dim='time')
print(f'Concatenated ZONs\n')

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

### Process data
ZON_data = ZONs.sel(time=slice(f'{epoch_start}-01-01', f'{epoch_end}-12-31'))
JVI_data = JVIs.sel(time=slice(f'{epoch_start}-01-01', f'{epoch_end}-12-31'))
max_ZON = round(ZON_data['ZON_AT'].max().item(), 5)
min_ZON = round(ZON_data['ZON_AT'].min().item(), 5)
max_JVI = round(JVI_data['JVI_AT'].max().item(), 5)
min_JVI = round(JVI_data['JVI_AT'].min().item(), 5)
print(f'ZON (min, max): ({min_ZON}, {max_ZON})')
print(f'JVI (min, max): ({min_JVI}, {max_JVI})\n')

### Bin size
bins = [np.arange(0, 901, 25), np.arange(10, 40.1, 1)]

### Colorbar
cmap = colormaps.get_cmap('viridis')
norm = BoundaryNorm(np.arange(0, 101, 5), ncolors=cmap.N, extend='max')

### Scatter
hist = plt.hist2d(ZON_data['ZON_AT'].values.flatten(), JVI_data['JVI_AT'].values.flatten(), bins=bins, cmap=cmap, cmin=1, norm=norm)
ticks = np.arange(0, 101, 25)
cbar = plt.colorbar(ticks=ticks, extend='max')
cbar.set_label('Days', fontsize=11, fontname='AvantGarde', rotation=270, labelpad=18)
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_yticklabels([f'{v:g}' for v in ticks])
cbar.minorticks_off()
cbar.ax.set_position([.83, .02, .025, .98])

### Title
plt.title(r'\textbf{ZON vs JVI}', color='k', fontsize=16, y=1.06)

### Ticks
plt.xticks(np.arange(0, 1001, 100), fontsize=8, fontname='Avant Garde')
plt.yticks(np.arange(10, 41, 5), fontsize=8, fontname='Avant Garde')
plt.tick_params(axis='both', labelcolor='k', color='dimgray')

### Labels
plt.xlabel(r'\textbf{ZON}', fontsize=12, color='k', labelpad=8); plt.ylabel(r'\textbf{JVI}', fontsize=12, color='k', labelpad=8)

### Annotations
plt.text(.5, 1.02, r'\textbf{Atlantic Sector (50°W–40°E)}', ha='center', fontsize=11, color='k', transform=plt.gca().transAxes)
plt.text(1, 1.01, f'{epoch_start}–{epoch_end}', color='k', fontsize=8, ha='right', transform=plt.gca().transAxes)

### Domain
plt.axis([0, 900, 15, 40])

### Save figure
ax.set_position([.15, .15, .65, .72])
plt.savefig(os.path.join(fig_dir, f'Reanalysis_AT_ZON_JVI_{epoch_start}–{epoch_end}.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')