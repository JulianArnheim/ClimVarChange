"""
Plot global mean Sea Surface Temperature (SST) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from statsmodels.nonparametric.smoothers_lowess import lowess
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
    ax.xaxis.set_ticks_position('bottom')

def cosine_weighted_spatial_mean(data, coord=None):
    coord = coord or ('lat' if 'lat' in data.coords else 'latitude' if 'latitude' in data.coords else None)
    if not coord:
        raise ValueError("No 'lat' or 'latitude' coordinate found in the data array.")
    if 'lat' in data.coords and 'latitude' in data.coords:
        raise ValueError("Both 'lat' and 'latitude' coordinates exist.")
    weights = np.cos(np.deg2rad(data[coord]))
    return (data * weights).sum(dim=coord) / (data.notnull() * weights).sum(dim=coord)

def custom_formatter(x, _):
    if abs(x) < .01:
        return '0'
    sign = '+' if x > 0 else '-'
    val = f'{abs(x):.1f}'.lstrip('0')
    return f'{sign}{val}'

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/SST/Global/')

### Years and epoch
year_min, year_max = 1940, 2025
epoch = (1981, 2010)

print(f'\n---- Sea Surface Temperature (SST) ----\n')

### Load data
file = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_SST_Monthly.nc')
datat_monthly = xr.open_dataset(file, chunks={'time': 12})

### Calculate annual means and anomalies
annual_datat = datat_monthly.groupby('time.year').mean(dim='time')
annual_mean_datat = cosine_weighted_spatial_mean(annual_datat.sum(dim='longitude')) / cosine_weighted_spatial_mean(annual_datat.notnull().sum(dim='longitude'))  # 2-D area mean over ocean points
refmean_datat = annual_mean_datat.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')
mean_datat_anom = annual_mean_datat - refmean_datat

### LOWESS smoothing
years = mean_datat_anom['year'].values
z_anom = mean_datat_anom['sst'].values
lowess_frac = 30 / (years[-1] - years[0])  # fraction ~ 30-year window
z_lowess = lowess(endog=z_anom, exog=years, frac=lowess_frac, return_sorted=False)

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', labelcolor='dimgray', facecolor='w')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure
fig, ax = plt.subplots(figsize=(10, 6))

### Adjust axes
adjust_spines(ax, ['left', 'bottom'])
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

### Ticks
plt.xticks(np.arange(1940, 2040, 10), np.arange(1940, 2040, 10), color='k')
plt.yticks(np.arange(-1, 1, .1), map(str, np.arange(-1, 1, .1)), color='k')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

### Domain
plt.xlim([1940, 2025])
plt.ylim([-.4, .7])

### Horizontal lines
for i in np.arange(-10, 11, .1):
    plt.axhline(i, color='gray', linewidth=1 if round(i, 1) == 0 else .5, linestyle='-' if round(i, 1) == 0 else 'dotted')

### Plot
plt.plot(years, z_anom, color='k', linewidth=3, label='Annual Mean')
plt.plot(years, z_lowess, color='deeppink', linewidth=3, label='30-Year LOWESS')

### Title
plt.title(r'\textbf{GLOBAL MEAN SEA SURFACE TEMPERATURE}', color='k', fontsize=22)

### Labels, annotation, and legend
plt.xlabel(r'\textbf{Year}', fontsize=16, color='k', labelpad=10)
plt.ylabel(r'\textbf{Sea Surface Temperature Anomaly [°C]}', fontsize=16, color='k', labelpad=10)
plt.text(2025, -.39, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}, at $\sim${refmean_datat["sst"].values:,.1f}°C}}', fontsize=11, ha='right', color='k')
l = plt.legend(shadow=False, fontsize=13, loc='upper center', bbox_to_anchor=(.5, 1), fancybox=True,
               ncol=2, frameon=False)

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_Global_SST_Anoms_{year_min}–{year_max}.png'), dpi=1000)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')