"""
Plot scatterplot between daily mid-latitude circulation diagnostics from ERA5 reanalysis

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

### Years and epoch
year_min, year_max = 1940, 2025
epoch_start, epoch_end = 1981, 2010

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
for i in np.arange(0, 1000, 25):
    plt.axvline(i, color='dimgray', linestyle='solid' if i % 100 == 0 else 'dotted', linewidth=.5, label='_nolegend_', zorder=3)
for i in np.arange(0, 5.5, .1):
    plt.axhline(i, color='dimgray', linestyle='solid' if np.isclose(i * 2, round(i * 2)) else 'dotted', linewidth=.5, label='_nolegend_', zorder=3)

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

### Process winter data
ZON_data = ZONs.sel(time=slice(f'{epoch_start - 1}-01-01', f'{epoch_end}-12-31')).where(ZONs.time.dt.month.isin([12, 1, 2]), drop=True)
SIN_data = SINs.sel(time=slice(f'{epoch_start - 1}-01-01', f'{epoch_end}-12-31')).where(SINs.time.dt.month.isin([12, 1, 2]), drop=True)
month, year = ZON_data.time.dt.month, ZON_data.time.dt.year
winter_year = xr.where(month == 12, year + 1, year)
ZON_data = ZON_data.assign_coords(winter_year=('time', winter_year.data))
SIN_data = SIN_data.assign_coords(winter_year=('time', winter_year.data))
valid_years = np.array([yr for yr in np.unique(winter_year) if len(np.unique(ZON_data.time.dt.month.where(ZON_data.winter_year == yr, drop=True))) == 3])
ZON_data = ZON_data.sel(time=ZON_data.winter_year.isin(valid_years))
SIN_data = SIN_data.sel(time=SIN_data.winter_year.isin(valid_years))

### Statistics
print(f'DJF winters: {valid_years[0]}–{valid_years[-1]} ({len(valid_years)} seasons)')
max_ZON, min_ZON = round(ZON_data['ZON_NH'].max().item(), 5), round(ZON_data['ZON_NH'].min().item(), 5)
max_SIN, min_SIN = round(SIN_data['SIN_NH'].max().item(), 5), round(SIN_data['SIN_NH'].min().item(), 5)
print(f'ZON (min, max): ({min_ZON}, {max_ZON})')
print(f'SIN (min, max): ({min_SIN}, {max_SIN})\n')

### Colorbar
cmap = colormaps.get_cmap('viridis')
norm = BoundaryNorm(np.arange(0, 151, 10), ncolors=cmap.N, extend='max')

### Scatter
hist = plt.hist2d(ZON_data['ZON_NH'].values.flatten(), SIN_data['SIN_NH'].values.flatten(), bins=[np.arange(200, 901, 25), np.arange(1, 5.1, .1)], cmap=cmap, cmin=1, norm=norm)
ticks = np.arange(0, 151, 50)
cbar = plt.colorbar(ticks=ticks, extend='max')
cbar.set_label('Days', fontsize=11, fontname='AvantGarde', rotation=270, labelpad=18)
cbar.ax.tick_params(labelsize=8)
cbar.ax.set_yticklabels([str(v) for v in ticks])
cbar.minorticks_off()
cbar.ax.set_position([.83, .02, .025, .98])

### Title
plt.title(r'\textbf{ZON vs SIN}', color='k', fontsize=16, y=1)

### Ticks
plt.xticks(np.arange(0, 1001, 100), fontsize=8, fontname='Avant Garde')
plt.yticks(np.arange(1, 2.6, .5), fontsize=8, fontname='Avant Garde')
plt.tick_params(axis='both', labelcolor='k', color='dimgray')

### Labels and annotations
plt.xlabel(r'\textbf{ZON}', fontsize=12, color='k', labelpad=8); plt.ylabel(r'\textbf{SIN}', fontsize=12, color='k', labelpad=8)
plt.text(1, 1.01, f'{epoch_start}–{epoch_end}', color='k', fontsize=8, ha='right', transform=plt.gca().transAxes)
plt.text(.01, .045, r'\textbf{DJF}', transform=ax.transAxes, fontsize=30, color='gray', fontweight='bold', ha='left', va='center', alpha=1, zorder=10)

### Domain
plt.axis([200, 700, 1, 2.5])

### Save figure
ax.set_position([.15, .15, .65, .72])
plt.savefig(os.path.join(fig_dir, f'Reanalysis_NH_ZON_SIN_DJF_{epoch_start}–{epoch_end}.png'), dpi=1000)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')