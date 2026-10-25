"""
Plot winter (DJF) histograms of the 850 hPa Jet Latitude Index (JLI) from White et
al. (2019) or Bracegirdle (2022) for several bin widths from ERA5 reanalysis

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

print(f'\n---- Jet Latitude Index (JLI) ----\n')

### Load data
JLI = xr.open_dataset(os.path.join(data_dir, f'ERA5_NA_JLI_JSI_{year_min}-{year_max}_Daily.nc'))['JLI_U850'].sel(time=slice(f'{year_min}-01-01', f'{year_max}-12-31'))
JLI = JLI.isel(time=JLI.time.dt.month.isin([12, 1, 2]))
JLI = JLI.assign_coords(winter_year=('time', xr.where(JLI.time.dt.month == 12, JLI.time.dt.year + 1, JLI.time.dt.year).data))
JLI = JLI.where((JLI.winter_year >= year_min + 1) & (JLI.winter_year <= year_max), drop=True)

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Loop through histogram bin sizes
for bin in range(1, 6):
    print(f'Plotting: {bin}/5')

    ### Create figure
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ### Adjust axes
    adjust_spines(ax, ['left', 'bottom'])
    for key in ['top', 'right']:
        ax.spines[key].set_color('none')
    for key in ['bottom', 'left']:
        ax.spines[key].set_linewidth(2)
    ax.tick_params(axis='both', length=5.5, width=2, which='major', colors='dimgray')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('k')

    ### Histogram and kernel density estimation
    plt.hist(JLI, bins=np.arange(15, 76, bin), orientation='vertical', color='gray', edgecolor='k', density=True, zorder=3)
    kde, kde_xs = gaussian_kde(JLI), np.linspace(15, 75, 10000)
    plt.plot(kde_xs, kde(kde_xs), color='k', zorder=5)

    ### Ticks and domain
    plt.xticks(np.arange(-90, 90, 5), [f'{deg}°' for deg in np.arange(-90, 90, 5)], color='k')
    plt.yticks(np.arange(0, .1, .01), ['0' if t == 0 else f'{t:.2f}'.lstrip('0') for t in np.arange(0, .1, .01)], color='k')
    plt.axis([15, 75, 0, .07])

    ### Title, labels, and annotation
    plt.title(r'\textbf{Jet Latitude Index (JLI)}', color='k', fontsize=20, y=1)
    plt.xlabel(r'\textbf{Latitude [°N]}', fontsize=13, color='k', labelpad=8); plt.ylabel(r'\textbf{Relative Frequency}', fontsize=13, color='k', labelpad=8)
    plt.text(.5, .93, f'DJF {year_min + 1}–{year_max}', color='k', fontsize=13, ha='center', transform=plt.gca().transAxes)

    ### Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_JLI_DJF_{year_min + 1}–{year_max}_bin{bin}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')