"""
Plot Arctic and global Sea Surface Temperature (SST) anomalies, "Arctic Amplification," from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.ticker as ticker
import numpy as np
import os
import time
import xarray as xr

mpl.use('Agg')  # non-interactive backend to avoid tkinter issues

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
    elif x > 0:
        return f'+{x:.1f}'.replace('+0', '+').replace('.0', '')
    else:
        return f'{x:.1f}'.replace('-0', '-').replace('.0', '')

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/SST/Arctic/')

### Years and epoch
year_min, year_max = 1940, 2025
epoch = (1981, 2010)

print(f'\n---- Arctic Amplification (SST) ----\n')

### Arctic definition
Arctic_min_lats = [60, 65, 66.5, 70]

### Load data
filename = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_SST_Monthly.nc')
datat = xr.open_dataset(filename, chunks={'time': 12})

### Loop through each Arctic definition
for Arctic_min_lat in Arctic_min_lats:

    ### Figure
    print(f'Arctic Minimum Latitude: {Arctic_min_lat}°N')

    ### Process data
    annual_datat = datat.sortby('time').groupby('time.year').mean(dim='time')

    ### Arctic annual means & anomalies
    arctic_data = annual_datat.where(annual_datat['latitude'] >= Arctic_min_lat, drop=True)
    annual_mean_Arctic = cosine_weighted_spatial_mean(arctic_data.sum(dim='longitude')) / cosine_weighted_spatial_mean(arctic_data.notnull().sum(dim='longitude'))  # 2-D area mean over ocean points
    Arctic_anom = annual_mean_Arctic - annual_mean_Arctic.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')

    ### Global annual means & anomalies
    annual_mean_global = cosine_weighted_spatial_mean(annual_datat.sum(dim='longitude')) / cosine_weighted_spatial_mean(annual_datat.notnull().sum(dim='longitude'))  # 2-D area mean over ocean points
    global_anom = annual_mean_global - annual_mean_global.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')

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
    fig, ax = plt.subplots(figsize=(6.5, 5))

    ### Adjust axes
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both', length=5.5, width=2, which='major')

    ### Ticks
    x_ticks = np.arange(1940, 2031, 10)
    y_ticks = np.arange(-10, 11, .1)
    plt.xticks(x_ticks, x_ticks, color='k')
    plt.yticks(y_ticks, map(str, y_ticks), color='k')

    ### Domain
    plt.xlim([1940, 2025]); plt.ylim([-.5, .8])

    ### Horizontal lines
    for step, style, lw in [(1, 'solid', .5), (.1, 'dotted', .5)]:
        for y in np.arange(-1, 1, step):
            plt.axhline(y, color='gray', linestyle=style, linewidth=1 if y == 0 else lw)

    ### Title
    plt.title(r'\textbf{ARCTIC AMPLIFICATION}', color='k', fontsize=25)

    ### Plot
    datasets = ['Global', rf'Arctic ($\geq${Arctic_min_lat}°N)']
    years = np.arange(year_min, year_max + 1, 1)
    plt.plot(years, global_anom['sst'], color='k', linewidth=3, label=rf'\textbf{{{datasets[0]}}}')
    plt.plot(years, Arctic_anom['sst'], color='crimson', linewidth=3, label=rf'\textbf{{{datasets[1]}}}')

    ### Labels, annotation, and legend
    plt.xlabel(r'\textbf{Year}', fontsize=13, color='k', labelpad=10)
    plt.ylabel(r'\textbf{Sea Surface Temperature Anomaly [°C]}', fontsize=13, color='k', labelpad=10)
    plt.text(2025, -.48, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}}}', fontsize=11, rotation='horizontal', ha='right', color='k')
    l = plt.legend(shadow=False, fontsize=11, loc='upper center', bbox_to_anchor=(.5, 1.015), fancybox=True, ncol=2, frameon=False)
    for text in l.get_texts():
        text.set_color('k')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

    ### Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_AA_SST_{year_min}–{year_max}_{Arctic_min_lat}N.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')