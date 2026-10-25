"""
Plot Arctic mean Sea Surface Temperature (SST) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib.colors import LinearSegmentedColormap
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
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/SST/Arctic/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Arctic Warming (SST) ----\n')

### Time specifications
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
month_positions = [1, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365]

### Arctic definition
Arctic_min_lat = 66.5

### Load data
datat = xr.open_dataset(os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_SST_Monthly.nc'))

### Process data
Arctic_datat = datat.sortby('time').where(datat['latitude'] >= Arctic_min_lat, drop=True)
Arctic_datat = cosine_weighted_spatial_mean(Arctic_datat.sum(dim='longitude')) / cosine_weighted_spatial_mean(Arctic_datat.notnull().sum(dim='longitude'))  # 2-D area mean over ocean points
Arctic_datat_by_year = Arctic_datat.groupby('time.year')

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

### Ticks
yticks = np.arange(-15, 15, 1)
ax.set_xticks(month_positions, month_labels, fontsize=12, color='k', rotation=0)
ax.set_yticks(yticks, map(str, yticks), color='k')

### Domain
ax.set_xlim(1, 365)
ax.set_ylim([-2, 3])

### Horizontal lines
for y in np.arange(-10, 11, 1):
    plt.axhline(y, color='gray', linestyle='solid' if y == 0 else 'dotted', linewidth=1 if y == 0 else .5)

### Plot
colors = ['#ffeda0', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000', '#000000']
years_sorted = np.unique(Arctic_datat['time.year'].values)
num_years = len(years_sorted)
blue_red_cmap = LinearSegmentedColormap.from_list('blue_red_cmap', colors, num_years)
for idx, year in enumerate(years_sorted):
    year_means = Arctic_datat.sel(time=str(year)).groupby('time.month').mean(dim='time')['sst'].values

    ### Start with months Jan–Dec
    full_year = year_means.copy()
    months = month_positions[:-1]

    ### Fetch previous December if it exists
    if idx > 0:
        prev_year_data = Arctic_datat.sel(time=str(years_sorted[idx - 1]))
        prev_dec = prev_year_data.sel(time=f'{years_sorted[idx - 1]}-12')['sst'].item()
        full_year = np.concatenate([[prev_dec], full_year])
        months = [month_positions[0] - 31] + months

    ### Fetch next January if it exists
    if idx < len(years_sorted) - 1:
        next_year_data = Arctic_datat.sel(time=str(years_sorted[idx + 1]))
        next_jan = next_year_data.sel(time=f'{years_sorted[idx + 1]}-01')['sst'].item()
        full_year = np.concatenate([full_year, [next_jan]])
        months = months + [month_positions[-1] + 31]

    plt.plot(months, full_year, color=blue_red_cmap(idx / (num_years - 1)))

### Title
plt.title(r'\textbf{ARCTIC MEAN SEA SURFACE TEMPERATURE}', color='k', fontsize=25)

### Labels
plt.xlabel(r'\textbf{Month}', fontsize=13, color='k', labelpad=10)
plt.ylabel(r'\textbf{Sea Surface Temperature [°C]}', fontsize=13, color='k', labelpad=10)

### Annotation
plt.text(360, -1.9, rf'\textbf{{Arctic: $\geq${Arctic_min_lat}°N}}', fontsize=11, rotation='horizontal', ha='right', color='k')

### Legend
decade_items = [(year, data) for year, data in Arctic_datat_by_year if year % 10 == 0]
l_handles = [plt.Line2D([0], [0], color=blue_red_cmap(i / (len(decade_items) - 1)), lw=2) for i, _ in enumerate(decade_items)]
l = plt.legend(handles=l_handles, labels=[f'{year}s' for year, _ in decade_items], shadow=False, fontsize=4, loc='upper left',
               bbox_to_anchor=(.12, .78), fancybox=True, ncol=3, frameon=False, prop={'weight': 'bold'}, labelspacing=.5)

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_Arctic_SST_{year_min}–{year_max}.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')