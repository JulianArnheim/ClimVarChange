"""
Plot global mean Sea Surface Temperature (SST) anomalies from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
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
    val = f'{abs(x):.1f}'
    if val.startswith('0'):
        val = val[1:]
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

### Time specifications
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
colors = ['#B0C4DE', '#9ACD32', '#FFC0CB',  '#D2691E', '#00FFFF', '#FFA500',
          '#FF69B4', '#32CD32', '#FFD700', '#800080', '#FF0000', '#7B68EE']

print(f'\n---- Global Warming (SST) ----\n')

### Load data
filename = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_SST_Monthly.nc')
datat = xr.open_dataset(filename, chunks={'time': 12})

### Process data
monthly_datat = datat.sortby('time')

### Global annual mean
global_datat = cosine_weighted_spatial_mean(monthly_datat.sum(dim='longitude')) / cosine_weighted_spatial_mean(monthly_datat.notnull().sum(dim='longitude'))  # 2-D area mean over ocean points
monthly_refmean = global_datat.sel(time=slice(f'{epoch[0]}-01-01', f'{epoch[1]}-12-31')).mean(dim='time')
annual_mean_anom = global_datat.groupby('time.year').mean(dim='time') - monthly_refmean

### Monthly temperature anomalies
global_datat_by_month = {}
for month, group in global_datat.groupby('time.month'):
    monthly_refmean = group.sel(time=slice(f'{epoch[0]}-01-01', f'{epoch[1]}-12-31'))['sst'].mean(dim='time')
    monthly_anom = group['sst'] - monthly_refmean
    group['monthly_anomaly'] = monthly_anom
    global_datat_by_month[month] = group

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
ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

### Ticks
x_ticks = np.arange(1940, 2026, 5)
y_ticks = np.arange(-10, 11, .1)
plt.xticks(x_ticks, x_ticks, color='k')
plt.yticks(y_ticks, map(str, y_ticks), color='k')

### Domain
plt.xlim([1940, 2025])
plt.ylim([-.5, .8])

### Horizontal lines
for step, style, lw in [(1, 'solid', .5), (.1, 'dotted', .5)]:
    for y in np.arange(-10, 10, step):
        plt.axhline(y, color='gray', linestyle=style, linewidth=1 if y == 0 else lw)

### Plot
for month, group in global_datat_by_month.items():
    plt.plot(group['time.year'], group['monthly_anomaly'], color=colors[month - 1], label=months[month - 1])
plt.plot(global_datat_by_month[1]['time.year'], annual_mean_anom['sst'].values, color='k', linewidth=3)

### Title
plt.title(r'\textbf{GLOBAL MEAN SEA SURFACE TEMPERATURE}', color='k', fontsize=25)

### Labels
plt.xlabel(r'\textbf{Year}', fontsize=13, color='k', labelpad=10)
plt.ylabel(r'\textbf{Sea Surface Temperature Anomaly [°C]}', fontsize=13, color='k', labelpad=10)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

### Annotation
plt.text(2025, -.48, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}}}', fontsize=11, ha='right', color='k')

### Legend
l = plt.legend(shadow=False, fontsize=11, loc='upper center', bbox_to_anchor=(.5, .992), fancybox=True, ncol=6, frameon=False)
for text in l.get_texts():
    text.set_color('k')

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_Global_SST_Anoms_Months_{year_min}–{year_max}.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')