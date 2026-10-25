"""
Plot Jet Latitude Index (JLI) and Jet Speed Index (JSI) from White
et al. (2019) or Bracegirdle (2022) for each decade of reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from scipy.stats import gaussian_kde
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
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Histograms/JLI/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Jet Latitude Index (JLI) and Jet Speed Index (JSI) ----\n')

### Load data
filename = os.path.join(data_dir, f'ERA5_NA_JLI_JSI_{year_min}-{year_max}_Daily.nc')
JLI = xr.open_dataset(filename)['JLI_U850'].sel(time=slice(f'{year_min}-12-01', f'{year_max}-02-28'))
JSI = xr.open_dataset(filename)['JSI_U850'].sel(time=slice(f'{year_min}-12-01', f'{year_max}-02-28'))

### Select DJF seasons
JLI = JLI.sel(time=JLI.time.dt.month.isin([12, 1, 2]))
JSI = JSI.sel(time=JSI.time.dt.month.isin([12, 1, 2]))
JLI = JLI.assign_coords(winter_year=('time', xr.where(JLI.time.dt.month == 12, JLI.time.dt.year + 1, JLI.time.dt.year).data))
JSI = JSI.assign_coords(winter_year=('time', xr.where(JSI.time.dt.month == 12, JSI.time.dt.year + 1, JSI.time.dt.year).data))
JLI = JLI.where((JLI.winter_year >= year_min + 1) & (JLI.winter_year <= year_max), drop=True)
JSI = JSI.where((JSI.winter_year >= year_min + 1) & (JSI.winter_year <= year_max), drop=True)

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.25, 4.5), gridspec_kw={'width_ratios': [6, 1]})

### Decades and colors
decades = [(start, min(start + 9, year_max)) for start in range(year_min, year_max + 1, 10)]
colors = plt.get_cmap('tab10').colors

### Build KDE and JSI dictionaries
KDE_dict, JSI_dict = {}, {}

### Legend labels
first_decade, last_decade = '1940s (9 seasons)', f'2020s (–{year_max})'
labels = [first_decade if s == 1940 else f'{s}s' if e != year_max else last_decade for s, e in decades]

### Loop through decades and compute KDE for JLI and mean for JSI
for (start, end), label_decade, color in zip(decades, labels, colors):

    ### Crop decade
    JLI_decade = JLI.where((JLI.winter_year >= start) & (JLI.winter_year <= end), drop=True)
    JSI_decade = JSI.where((JSI.winter_year >= start) & (JSI.winter_year <= end), drop=True)

    ### Kernel density estimation
    kde_xs = np.linspace(15, 75, 10000)
    kde_vals = gaussian_kde(JLI_decade.values)(kde_xs)

    ### Store KDE and JSI
    KDE_dict[label_decade] = {'x': kde_xs, 'y': kde_vals, 'color': color}
    JSI_dict[label_decade] = {'speed': float(JSI_decade.mean()), 'color': color}

#############################################
### Left panel : Jet Latitude Index (JLI) ###
#############################################

### Plot KDEs
for label_decade in KDE_dict:
    ax1.plot(KDE_dict[label_decade]['x'], KDE_dict[label_decade]['y'], label=label_decade, color=KDE_dict[label_decade]['color'], linewidth=2, zorder=2)

### Adjust axes
adjust_spines(ax1, ['left', 'bottom'])
for key in ['top', 'right']:
    ax1.spines[key].set_color('none')
for key in ['bottom', 'left']:
    ax1.spines[key].set_linewidth(2)
ax1.tick_params(axis='both', length=5.5, width=2, which='major', colors='dimgray')
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_color('k')

### Title
ax1.set_title(r'\textbf{Jet Latitude Index (JLI)}', color='k', fontsize=20, y=1)

### Ticks
lat_ticks = np.arange(-90, 90, 5)
ax1.set_xticks(lat_ticks)
ax1.set_xticklabels([f'{deg}°' for deg in lat_ticks], color='k')
ax1.set_yticks(np.arange(0, .1, .01))
ax1.set_yticklabels(['0' if t == 0 else f'{t:.2f}'.lstrip('0') for t in np.arange(0, .1, .01)], color='k')

### Labels
ax1.set_xlabel(r'\textbf{Latitude [°N]}', fontsize=13, color='k', labelpad=8)
ax1.set_ylabel(r'\textbf{Relative Frequency}', fontsize=13, color='k', labelpad=8)

### Annotation
ax1.text(.5, .95, f'DJF {year_min + 1}–{year_max}', color='k', fontsize=13, ha='center', transform=ax1.transAxes)

### Domain
ax1.set(xlim=[15, 75], ylim=[0, .06])

### Legend
ax1.legend(loc='upper left', bbox_to_anchor=(.03, .91), frameon=False, fontsize=7.5, handlelength=1.2, labelspacing=.3)

### Letter
ax1.text(-.19, 1.1, r'\textbf{(a)}', transform=ax1.transAxes, fontsize=11, ha='left', va='top')

###########################################
### Right panel : Jet Speed Index (JSI) ###
###########################################

### Plot JSI
for label_decade in JSI_dict:
    ax2.scatter(0, JSI_dict[label_decade]['speed'], color=JSI_dict[label_decade]['color'], marker='x', s=30, linewidth=2, zorder=3)

### Adjust axes
adjust_spines(ax2, ['left'])
for key in ['top', 'right', 'bottom']:
    ax2.spines[key].set_color('none')
ax2.spines['left'].set_linewidth(2)
ax2.tick_params(axis='y', length=5.5, width=2, which='major', colors='dimgray')
for label in ax2.get_yticklabels():
    label.set_color('k')

### Ticks
ax2.set_xticks([])
ax2.set_yticks(np.arange(10, 20, 1))
ax2.set_yticklabels([f'{t}' for t in np.arange(10, 20, 1)], color='k')

### Label
ax2.set_ylabel(r'\textbf{Jet Speed Index (JSI) [m s\textsuperscript{{-1}}]}', fontsize=13, color='k', labelpad=8)

### Domain
ax2.set(xlim=[-.01, .01], ylim=[14, 17])

### Letter
ax2.text(-1.1, 1.1, r'\textbf{(b)}', transform=ax2.transAxes, fontsize=11, ha='left', va='top')

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_JLI_JSI_DJF_{year_min + 1}–{year_max}_Decades.png'), dpi=1000)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')