"""
Plot Jet Latitude Index (JLI) and Jet Speed Index (JSI) from White et al. (2019) or
Bracegirdle (2022) for different phase categories of the North Atlantic Oscillation (NAO)

Notes
-----
    Data : (ERA5) https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Data : (NOAA) https://psl.noaa.gov/data/timeseries/daily/NAO/
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

def full_DJF(ds):
    ds = ds.assign_coords(winter_year=('time', xr.where(ds.time.dt.month == 12, ds.time.dt.year + 1, ds.time.dt.year).data))
    full_djf = [y for y in np.unique(ds.winter_year.values) if set(ds.time.dt.month.where(ds.winter_year == y, drop=True).values) == {12, 1, 2}]
    return ds.where(ds.winter_year.isin(full_djf), drop=True)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Histograms/JLI/')

### Years
year_min, year_max = 1948, 2025

print(f'\n---- Jet Latitude Index (JLI) and Jet Speed Index (JSI) ----\n')

### Load and process daily NAO index
filename = os.path.join(data_dir, 'PSL_NAO_1948-2025_Daily.nc')
NAO_ds = xr.open_dataset(filename).sel(time=slice(f'{year_min - 1}-12-01', f'{year_max}-02'))
NAO_ds = NAO_ds.isel(time=NAO_ds.time.dt.month.isin([12, 1, 2]))
NAO_ds = full_DJF(NAO_ds)

### NAO categories
num = 1000  # number of extreme days
NAO_sorted = NAO_ds['NAO'].sortby(NAO_ds['NAO'])
NAO_high, NAO_low = NAO_sorted[-num:].sortby('time'), NAO_sorted[:num].sortby('time')
NAO_mid = NAO_ds.sel(time=NAO_sorted.isel(time=slice(len(NAO_sorted) // 2 - num // 2, len(NAO_sorted) // 2 - num // 2 + num)).time)

### Load data
filename = os.path.join(data_dir, f'ERA5_NA_JLI_JSI_1940-{year_max}_Daily.nc')
JLI_ds = xr.open_dataset(filename).sel(time=slice(f'{year_min - 1}-12-01', f'{year_max}-02'))
JLI_ds = JLI_ds.isel(time=JLI_ds.time.dt.month.isin([12, 1, 2]))
JLI_ds = JLI_ds.assign_coords(winter_year=('time', xr.where(JLI_ds.time.dt.month == 12, JLI_ds.time.dt.year + 1, JLI_ds.time.dt.year).data))
JLI_ds = full_DJF(JLI_ds)
JLI, JSI = (JLI_ds[var] for var in ['JLI_U850', 'JSI_U850'])
first_winter, last_winter = int(JLI_ds.winter_year.min()), int(JLI_ds.winter_year.max())

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 4.5), gridspec_kw={'width_ratios': [6, 1.2]})

#############################################
### Left panel : Jet Latitude Index (JLI) ###
#############################################

### Adjust axes
adjust_spines(ax1, ['left', 'bottom'])
for key in ['top', 'right']:
    ax1.spines[key].set_color('none')
for key in ['bottom', 'left']:
    ax1.spines[key].set_linewidth(2)
ax1.tick_params(axis='both', length=5.5, width=2, which='major', colors='dimgray')
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_color('k')

### Plot KDE of JLI by NAO category
kde_xs = np.linspace(15, 75, 10000)
NAO_categories = [('Highest', NAO_high, 'orange'), ('Middle', NAO_mid, 'gray'), ('Lowest', NAO_low, 'steelblue')]
for label, NAO_data, color in NAO_categories:
    latitudes = np.array(JLI.where(JLI.time.isin(NAO_data.time), drop=True))
    kde = gaussian_kde(latitudes)
    ax1.plot(kde_xs, kde(kde_xs), color=color, linewidth=2.5, zorder=1, label=f'{label} {num} NAO days')

### Plot KDE for all days
ax1.plot(kde_xs, gaussian_kde(np.array(JLI))(kde_xs), color='k', linestyle=(0, (2, 1)), linewidth=3, zorder=2, label=f'All days ($n$ = {len(JLI.time)})')

### Ticks and domain
ax1.set_xticks(np.arange(-90, 90, 5))
ax1.set_xticklabels([f'{deg}°' for deg in np.arange(-90, 90, 5)], color='k')
ax1.set_yticks(np.arange(0, .1, .01))
ax1.set_yticklabels(['0' if t == 0 else f'{t:.2f}'.lstrip('0') for t in np.arange(0, .1, .01)], color='k')
ax1.set(xlim=[15, 75], ylim=[0, .09])

### Title, labels, and annotation
ax1.set_title(r'\textbf{Jet Latitude Index (JLI)}', color='k', fontsize=20, y=1)
ax1.set_xlabel(r'\textbf{Latitude [°N]}', fontsize=13, color='k', labelpad=8); ax1.set_ylabel(r'\textbf{Relative Frequency}', fontsize=13, color='k', labelpad=8)
ax1.text(.5, .95, f'DJF {first_winter}–{last_winter}', color='k', fontsize=13, ha='center', transform=ax1.transAxes)

### Legend
ax1.legend(loc='upper left', bbox_to_anchor=(0, .93), frameon=False, ncol=1, fontsize=8, handlelength=2.75)

### Letter
ax1.text(-.19, 1.1, r'\textbf{(a)}', transform=ax1.transAxes, fontsize=11, ha='left', va='top')

#######################################################
### Right panel : Jet Speed Index (JSI) by NAO days ###
#######################################################

### Adjust axes
adjust_spines(ax2, ['left'])
for key in ['top', 'right', 'bottom']:
    ax2.spines[key].set_color('none')
ax2.spines['left'].set_linewidth(2)
ax2.tick_params(axis='y', length=5.5, width=2, which='major', colors='dimgray')
for label in ax2.get_yticklabels():
    label.set_color('k')

### Violin plots
violin_data, violin_colors = [], []
for label, NAO_data, color in NAO_categories:
    jsi = np.array(JSI.where(JSI.time.isin(NAO_data.time), drop=True))
    violin_data.append(jsi)
    violin_colors.append(color)
vp = ax2.violinplot(violin_data, positions=[1, 2, 3], widths=.6, showmeans=False, showmedians=True, showextrema=True)
for i, body in enumerate(vp['bodies']):
    body.set(facecolor=violin_colors[i], edgecolor='k', linewidth=1, alpha=1)
vp['cmedians'].set(color='k', linewidth=1.5)
vp['cmins'].set(color='k', linewidth=1)
vp['cmaxes'].set(color='k', linewidth=1)
vp['cbars'].set(color='k', linewidth=1)

### Mean JSI and quartiles (Highest, Middle, Lowest NAO) [TL, TR, BL, BR]
quartile_offsets = [[.25, .25, .25, .25], [.25, .25, .27, .26], [.22, .22, .26, .25]]
for i, jsi in enumerate(violin_data, start=1):
    ax2.scatter(i, np.mean(jsi), color='k', marker='x', s=25, linewidth=1, zorder=51)
    tl, tr, bl, br = quartile_offsets[i - 1]
    ax2.hlines(np.percentile(jsi, 25), i - bl, i + br, color='k', linewidth=1.25, linestyle=(0, (2, 1)), zorder=2)
    ax2.hlines(np.percentile(jsi, 75), i - tl, i + tr, color='k', linewidth=1.25, linestyle=(0, (2, 1)), zorder=2)

### Ticks, label, and domain
ax2.set_xticks([]); ax2.set_yticks(np.arange(0, 50, 5))
ax2.set_yticklabels([f'{t}' for t in np.arange(0, 50, 5)], color='k')
ax2.set_ylabel(r'\textbf{Jet Speed Index (JSI) [m s\textsuperscript{-1}]}', fontsize=13, color='k', labelpad=8)
ax2.set(xlim=[.5, 3.5], ylim=[0, 35])

### Letter
ax2.text(-.95, 1.1, r'\textbf{(b)}', transform=ax2.transAxes, fontsize=11, ha='left', va='top')

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_JLI_JSI_DJF_{first_winter}–{last_winter}_NAO.png'), dpi=1000)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')