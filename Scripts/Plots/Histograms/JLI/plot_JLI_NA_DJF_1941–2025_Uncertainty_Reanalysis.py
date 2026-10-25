"""
Plot Jet Latitude Index (JLI) and Jet Speed Index (JSI) from White et al. (2019) or
Bracegirdle (2022) distribution with uncertainty via bootstrapping for reanalysis

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
ds = xr.open_dataset(filename).sel(time=slice(f'{year_min}-01-01', f'{year_max}-12-31'))
ds = ds.isel(time=ds.time.dt.month.isin([12, 1, 2]))
ds = ds.assign_coords(winter_year=('time', xr.where(ds.time.dt.month == 12, ds.time.dt.year + 1, ds.time.dt.year).data))
ds = ds.where((ds.winter_year >= year_min + 1) & (ds.winter_year <= year_max), drop=True)
JLI, JSI = ds['JLI_U850'], ds['JSI_U850']

print(f'Uncertainty via bootstrapping (30 random years with replacement):')

### Bootstrap settings
num_iterations = 1000
rng = np.random.default_rng(42)

### Bootstrap resampling for uncertainty
kde_xs = np.linspace(15, 75, 10000)
years = np.arange(year_min + 1, year_max + 1)
JLI_yr = {y: JLI.where(JLI.winter_year == y, drop=True).values for y in years}
JSI_yr = {y: JSI.where(JSI.winter_year == y, drop=True).values for y in years}
pdf_samples, jsi_samples = [], []
for i in range(num_iterations):

    ### Sample 30 years of data (with replacement) and concatenate JLI and JSI values for those years
    sample_years = rng.choice(years, size=30, replace=True)
    sample_JLI = np.concatenate([JLI_yr[y] for y in sample_years])
    sample_JSI = np.concatenate([JSI_yr[y] for y in sample_years])

    ### Append KDE and mean JSI
    kde = gaussian_kde(sample_JLI)
    pdf_samples.append(kde(kde_xs))
    jsi_samples.append(np.mean(sample_JSI))

    ### Progress
    if (i + 1) % 100 == 0:
        print(f'   Iteration {i + 1}/{num_iterations} completed\n')

### Compute standard deviation of PDFs to estimate confidence intervals
pdf_std = np.std(pdf_samples, axis=0)

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

### Adjust axes
adjust_spines(ax1, ['left', 'bottom'])
for key in ['top', 'right']:
    ax1.spines[key].set_color('none')
for key in ['bottom', 'left']:
    ax1.spines[key].set_linewidth(2)
ax1.tick_params(axis='both', length=5.5, width=2, which='major', colors='dimgray')
for label in ax1.get_xticklabels() + ax1.get_yticklabels():
    label.set_color('k')

### Plot KDE of the full record and bootstrap uncertainty spread
kde = gaussian_kde(np.ravel(JLI.values))
ax1.plot(kde_xs, kde(kde_xs), color='k', linewidth=2, zorder=1)
ax1.fill_between(kde_xs, kde(kde_xs) - pdf_std, kde(kde_xs) + pdf_std, color='gray', linewidth=0, edgecolor='none', zorder=0)

### Title
ax1.set_title(r'\textbf{Jet Latitude Index (JLI)}', color='k', fontsize=20, y=1)

### Ticks and domain
ax1.set_xticks(np.arange(-90, 90, 5))
ax1.set_xticklabels([f'{deg}°' for deg in np.arange(-90, 90, 5)], color='k')
ax1.set_yticks(np.arange(0, .1, .01))
ax1.set_yticklabels(['0' if t == 0 else f'{t:.2f}'.lstrip('0') for t in np.arange(0, .1, .01)], color='k')
ax1.set(xlim=[15, 75], ylim=[0, .06])

### Labels and annotations
ax1.set_xlabel(r'\textbf{Latitude [°N]}', fontsize=13, color='k', labelpad=8)
ax1.set_ylabel(r'\textbf{Relative Frequency}', fontsize=13, color='k', labelpad=8)
ax1.text(.5, .95, f'DJF {year_min + 1}–{year_max}', color='k', fontsize=13, ha='center', transform=ax1.transAxes)
ax1.text(.5, .9, r'$\pm$1 standard deviation', color='k', fontsize=8, ha='center', transform=ax1.transAxes)

### Letter
ax1.text(-.19, 1.1, r'\textbf{(a)}', transform=ax1.transAxes, fontsize=11, ha='left', va='top')

###########################################
### Right panel : Jet Speed Index (JSI) ###
###########################################

### Adjust axes
adjust_spines(ax2, ['left'])
for key in ['top', 'right', 'bottom']:
    ax2.spines[key].set_color('none')
ax2.spines['left'].set_linewidth(2)
ax2.tick_params(axis='y', length=5.5, width=2, which='major', colors='dimgray')
for label in ax2.get_yticklabels():
    label.set_color('k')

### Violin plot
vp = ax2.violinplot(JSI, positions=[1], widths=.3, showmeans=False, showmedians=True, showextrema=True)
for body in vp['bodies']:
    body.set(facecolor='lightgray', edgecolor='k', linewidth=1, alpha=1)
vp['cmedians'].set(color='k', linewidth=1.5)
vp['cmins'].set(color='k', linewidth=1)
vp['cmaxes'].set(color='k', linewidth=1)
vp['cbars'].set(color='k', linewidth=1)

### Mean JSI and quartiles
ax2.scatter(1, np.mean(JSI), color='k', marker='x', s=40, linewidth=2, zorder=1)
ax2.hlines([np.percentile(JSI, 25), np.percentile(JSI, 75)], .87, 1.13, color='k', linewidth=1.25, linestyle=(0, (2, 1)), zorder=2)

### Ticks and domain
ax2.set_xticks([]); ax2.set_yticks(np.arange(0, 50, 5))
ax2.set_yticklabels([f'{t}' for t in np.arange(0, 50, 5)], color='k')
ax2.set(xlim=[.7, 1.3], ylim=[0, 35])

### Label
ax2.set_ylabel(r'\textbf{Jet Speed Index (JSI) [m s\textsuperscript{-1}]}', fontsize=13, color='k', labelpad=8)

### Letter
ax2.text(-1.1, 1.1, r'\textbf{(b)}', transform=ax2.transAxes, fontsize=11, ha='left', va='top')

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_JLI_JSI_DJF_{year_min + 1}–{year_max}_Uncertainty.png'), dpi=1000)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')