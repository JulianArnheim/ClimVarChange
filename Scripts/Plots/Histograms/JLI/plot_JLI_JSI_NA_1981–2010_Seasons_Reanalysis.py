"""
Plot Jet Latitude Index (JLI) and Jet Speed Index (JSI) from White et
al. (2019) or Bracegirdle (2022) by season for reanalysis

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
year_min, year_max, (epoch_1, epoch_2) = 1940, 2025, (1981, 2010)

print(f'\n---- Jet Latitude Index (JLI) and Jet Speed Index (JSI) ----\n')

### Seasons and colors
seasons = {'DJF': ('Winter', 'dodgerblue'), 'MAM': ('Spring', 'limegreen'), 'JJA': ('Summer', 'orange'), 'SON': ('Autumn', 'indianred')}

### Load data
filename = os.path.join(data_dir, f'ERA5_NA_JLI_JSI_{year_min}-{year_max}_Daily.nc')
JLI, JSI = [xr.open_dataset(filename)[var].sel(time=slice(f'{epoch_1 - 1}-12-01', f'{epoch_2}-11-30')) for var in ['JLI_U850', 'JSI_U850']]

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.25, 4.5), gridspec_kw={'width_ratios': [6, 1]})

### Loop through each season
KDE_dict, JSI_dict = {}, {}
for season, (label, color) in seasons.items():

    ### Seasonal subsets
    JLI_season, JSI_season = JLI.sel(time=(JLI['time.season'] == season)), JSI.sel(time=(JSI['time.season'] == season))

    ### Kernel density estimation
    kde_xs = np.linspace(15, 75, 10000)
    kde_vals = gaussian_kde(JLI_season.values)(kde_xs)

    ### Store KDE and JSI values
    KDE_dict[season] = {'label': label, 'color': color, 'x': kde_xs, 'y': kde_vals}
    JSI_dict[season] = {'color': color, 'speed': float(JSI_season.mean())}

### Add all-seasons JSI
JSI_dict['All'] = {'color': 'k', 'speed': float(JSI.mean())}

### Kernel density estimation for all seasons combined
KDE_all = {'x': kde_xs, 'y': gaussian_kde(JLI.values)(kde_xs)}

#############################################
### Left panel : Jet Latitude Index (JLI) ###
#############################################

### Plot KDEs for each season and all seasons combined
for season in seasons:
    ax1.plot(KDE_dict[season]['x'], KDE_dict[season]['y'], label=f"{KDE_dict[season]['label']} ({season})", color=KDE_dict[season]['color'], zorder=2)
ax1.plot(KDE_all['x'], KDE_all['y'], color='k', linewidth=3, zorder=3)

### Adjust axes
adjust_spines(ax1, ['left', 'bottom'])
for key in ['top', 'right']:
    ax1.spines[key].set_color('none')
for key in ['bottom', 'left']:
    ax1.spines[key].set_linewidth(2)
ax1.tick_params(axis='both', length=5.5, width=2, which='major', colors='dimgray')
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_color('k')

### Title, ticks, labels, annotation, domain, and legend
ax1.set_title(r'\textbf{Jet Latitude Index (JLI)}', color='k', fontsize=20, y=1)
ax1.set_xticks(np.arange(-90, 90, 5))
ax1.set_xticklabels([f'{deg}°' for deg in np.arange(-90, 90, 5)], color='k')
ax1.set_yticks(np.arange(0, .1, .01))
ax1.set_yticklabels(['0' if t == 0 else f'{t:.2f}'.lstrip('0') for t in np.arange(0, .1, .01)], color='k')
ax1.set_xlabel(r'\textbf{Latitude [°N]}', fontsize=13, color='k', labelpad=8)
ax1.set_ylabel(r'\textbf{Relative Frequency}', fontsize=13, color='k', labelpad=8)
ax1.text(.5, .95, f'{epoch_1}–{epoch_2}', color='k', fontsize=13, ha='center', transform=ax1.transAxes)
ax1.set(xlim=[15, 75], ylim=[0, .07])
ax1.legend(loc='upper left', bbox_to_anchor=(.04, .88), frameon=False, fontsize=8, handlelength=1.2)

### Letter
ax1.text(-.19, 1.1, r'\textbf{(a)}', transform=ax1.transAxes, fontsize=11, ha='left', va='top')

###########################################
### Right panel : Jet Speed Index (JSI) ###
###########################################

### Plot jet speed for each season and all seasons combined
for season in seasons:
    ax2.scatter(0, JSI_dict[season]['speed'], color=JSI_dict[season]['color'], marker='x', s=30, linewidth=2, zorder=3)
ax2.scatter(0, JSI_dict['All']['speed'], color=JSI_dict['All']['color'], marker='x', s=50, linewidth=2, zorder=4)

### Adjust axes
adjust_spines(ax2, ['left'])
for key in ['top', 'right', 'bottom']:
    ax2.spines[key].set_color('none')
ax2.spines['left'].set_linewidth(2)
ax2.tick_params(axis='y', length=5.5, width=2, which='major', colors='dimgray')
for label in ax2.get_yticklabels():
    label.set_color('k')

### Ticks, label, and domain
ax2.set_xticks([])
ax2.set_yticks(np.arange(10, 20, 1)); ax2.set_yticklabels([f'{t}' for t in np.arange(10, 20, 1)], color='k')
ax2.set_ylabel(r'\textbf{Jet Speed Index (JSI) [m s\textsuperscript{{-1}}]}', fontsize=13, color='k', labelpad=8)
ax2.set(xlim=[-.01, .01], ylim=[10, 16])

### Letter
ax2.text(-1.1, 1.1, r'\textbf{(b)}', transform=ax2.transAxes, fontsize=11, ha='left', va='top')

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_JLI_JSI_{epoch_1}–{epoch_2}_Seasons.png'), dpi=1000)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')