"""
Plot Jet Latitude Index (JLI) and Jet Speed Index (JSI)
from White et al. (2019) or Bracegirdle (2022) for different pressure levels

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
year_min, year_max = 1981, 2010

### Pressure levels and colors
levels = [700, 750, 800, 850, 900, 925]
colors = plt.cm.viridis(np.linspace(.15, .9, len(levels)))

print(f'\n---- Jet Latitude Index (JLI) and Jet Speed Index (JSI) ----\n')

### Load data
filename = os.path.join(data_dir, 'ERA5_NA_JLI_JSI_1940-2025_Daily.nc')
ds = xr.open_dataset(filename).sel(time=slice(f'{year_min - 1}-12-01', f'{year_max}-12-31'))
ds = ds.isel(time=ds.time.dt.month.isin([12, 1, 2]))
ds = ds.assign_coords(winter_year=('time', xr.where(ds.time.dt.month == 12, ds.time.dt.year + 1, ds.time.dt.year).data))
ds = ds.where((ds.winter_year >= year_min) & (ds.winter_year <= year_max), drop=True)

### Build dictionaries
KDE_dict, JSI_dict, kde_xs = {}, {}, np.linspace(15, 75, 10000)
for lev, color in zip(levels, colors):

    ### Extract level-specific variables, remove NaNs, run kernel density estimation
    JLI_lev, JSI_lev = ds[f'JLI_U{lev}'].values, ds[f'JSI_U{lev}'].values
    JLI_lev, JSI_lev = JLI_lev[np.isfinite(JLI_lev)], JSI_lev[np.isfinite(JSI_lev)]
    kde_vals = gaussian_kde(JLI_lev)(kde_xs)

    ### Store
    KDE_dict[lev] = {'x': kde_xs, 'y': kde_vals, 'color': color}
    JSI_dict[lev] = {'speed': float(np.nanmean(JSI_lev)), 'color': color}

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.25, 4.5), gridspec_kw={'width_ratios': [6, 1]})

#############################################
### Left panel : Jet Latitude Index (JLI) ###
#############################################

### Plot KDEs
for lev in levels:
    ax1.plot(KDE_dict[lev]['x'], KDE_dict[lev]['y'], color=KDE_dict[lev]['color'], linewidth=2, label=f'{lev} hPa', zorder=2)

### Adjust axes
adjust_spines(ax1, ['left', 'bottom'])
for key in ['top', 'right']:
    ax1.spines[key].set_color('none')
for key in ['bottom', 'left']:
    ax1.spines[key].set_linewidth(2)
ax1.tick_params(axis='both', length=5.5, width=2, which='major', colors='dimgray')
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_color('k')

### Ticks and domain
ax1.set_xticks(np.arange(-90, 90, 5))
ax1.set_xticklabels([f'{deg}°' for deg in np.arange(-90, 90, 5)], color='k')
ax1.set_yticks(np.arange(0, .1, .01))
ax1.set_yticklabels(['0' if t == 0 else f'{t:.2f}'.lstrip('0') for t in np.arange(0, .1, .01)], color='k')
ax1.set(xlim=[15, 75], ylim=[0, .06])

### Title, labels, and annotation
ax1.set_title(r'\textbf{Jet Latitude Index (JLI)}', color='k', fontsize=20, y=1)
ax1.set_xlabel(r'\textbf{Latitude [°N]}', fontsize=13, color='k', labelpad=8)
ax1.set_ylabel(r'\textbf{Relative Frequency}', fontsize=13, color='k', labelpad=8)
ax1.text(.5, .95, f'DJF {year_min}–{year_max}', color='k', fontsize=13, ha='center', transform=ax1.transAxes)

### Legend
ax1.legend(loc='upper left', bbox_to_anchor=(.04, .88), frameon=False, fontsize=8, handlelength=1.2)

### Letter
ax1.text(-.19, 1.1, r'\textbf{(a)}', transform=ax1.transAxes, fontsize=11, ha='left', va='top')

###########################################
### Right panel : Jet Speed Index (JSI) ###
###########################################

### Plot JSI
for lev in levels:
    ax2.scatter(0, JSI_dict[lev]['speed'], color=JSI_dict[lev]['color'], marker='x', s=30, linewidth=2, zorder=3)

### Adjust axes
adjust_spines(ax2, ['left'])
for key in ['top', 'right', 'bottom']:
    ax2.spines[key].set_color('none')
ax2.spines['left'].set_linewidth(2)
ax2.tick_params(axis='y', length=5.5, width=2, which='major', colors='dimgray')
for label in ax2.get_yticklabels():
    label.set_color('k')

### Ticks and domain
ax2.set_xticks([]); ax2.set_yticks(np.arange(10, 25, 1))
ax2.set_yticklabels([f'{t}' for t in np.arange(10, 25, 1)], color='k')
ax2.set(xlim=[-.01, .01], ylim=[13, 20])

### Label
ax2.set_ylabel(r'\textbf{Jet Speed Index (JSI) [m s\textsuperscript{-1}]}', fontsize=13, color='k', labelpad=8)

### Letter
ax2.text(-1.1, 1.1, r'\textbf{(b)}', transform=ax2.transAxes, fontsize=11, ha='left', va='top')

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_JLI_JSI_DJF_{year_min}–{year_max}_Levels.png'), dpi=1000)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')