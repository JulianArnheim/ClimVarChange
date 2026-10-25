"""
Plot Arctic annual mean Sea Ice Volume (SIV) anomalies from the Pan-Arctic Ice Modeling and Assimilation System (PIOMAS) reanalysis

Notes
-----
    Data : https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/
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

def custom_formatter(x, _):
    if abs(x) < .01:
        return '0'
    sign = '+' if x > 0 else '-'
    val = f'{abs(x):.1f}'.lstrip('0')
    if val.endswith('.0'):
        val = val[:-2]
    return f'{sign}{val}'

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/ICEFRAC/SIV/')

### Years
year_crop = 1980
year_min, year_max = 1979, 2025

### Epoch
epoch = (1981, 2010)

print(f'\n---- Arctic Sea Ice Volume ----\n')

### Load data
filename = os.path.join(data_dir, f'PIOMAS_Arctic_{year_min}-{year_max}_SIV_Monthly.nc')
siv = xr.open_dataset(filename)['SIV'].sortby('time')

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', labelcolor='dimgray', facecolor='w')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Calculate annual mean sea ice volume
siv_annual = siv.groupby('time.year').mean(dim='time')

### Calculate sea ice volume anomalies
baseline_volume = siv_annual.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')
siv_anom = siv_annual - baseline_volume

### LOWESS smoothing
years, z_anom = siv_anom['year'].values, siv_anom.values
lowess_frac = 30 / (years[-1] - years[0])  # fraction ~ 30-year window
z_lowess = lowess(endog=z_anom, exog=years, frac=lowess_frac, return_sorted=False)

### Create figure
fig, ax = plt.subplots(figsize=(10, 6))

### Adjust axes
adjust_spines(ax, ['left', 'bottom'])
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

### Ticks
x_ticks = np.arange(1940, 2040, 10)
y_ticks = np.arange(-10, 5.1, 5)
plt.xticks(x_ticks, x_ticks, color='k')
plt.yticks(y_ticks, map(str, y_ticks), color='k')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

### Domain
ax.axis([year_crop, year_max, -10, 5])

### Horizontal lines
for i in np.arange(-10, 11, 1):
    plt.axhline(i, color='gray', linewidth=1 if i == 0 else .5, linestyle='-' if i % 5 == 0 else 'dotted')

### Plot
plt.plot(years, z_anom, color='k', linewidth=3, label=f'Annual Mean'); plt.plot(years, z_lowess, color='deeppink', linewidth=3, label='30-Year LOWESS')

### Title
plt.title(rf'\textbf{{ARCTIC SEA ICE VOLUME}}', color='k', fontsize=30)

### Labels
plt.xlabel(r'\textbf{Year}', fontsize=16, color='k', labelpad=10)
plt.ylabel(r'\textbf{Sea Ice Volume Anomaly [10\textsuperscript{{3}} km\textsuperscript{{3}}]}', fontsize=16, color='k', labelpad=10)

### Annotation
plt.text(2025, -9.85, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}, {baseline_volume * 1e3:,.0f} km\textsuperscript{{3}}}}', fontsize=11, ha='right', color='k')

### Legend
plt.legend(shadow=False, fontsize=13, loc='upper center', bbox_to_anchor=(.5, 1.015), fancybox=True, ncol=2, frameon=False)

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_PIOMAS_SIV_Anoms_{year_crop}–{year_max}.png'), dpi=1000)
plt.close(fig)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')