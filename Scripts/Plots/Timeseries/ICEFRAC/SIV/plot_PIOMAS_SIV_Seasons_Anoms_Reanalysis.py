"""
Plot Arctic seasonal Sea Ice Volume (SIV) anomalies from the Pan-Arctic Ice Modeling and Assimilation System (PIOMAS) reanalysis

Notes
-----
    Data : https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/
    Author : Julian B Arnheim
"""

### Import modules
from scipy.stats import linregress
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

def custom_formatter(x, _):
    if abs(x) < .01:
        return '0'
    sign = '+' if x > 0 else ''
    return f'{sign}{x:.1f}'.rstrip('0').rstrip('.').replace('0.', '.')

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/ICEFRAC/SIV/')

### Years and epoch
year_crop = 1980
year_min, year_max = 1979, 2025
years, epoch = np.arange(year_crop, year_max + 1), (1981, 2010)

print(f'\n---- Arctic Sea Ice Volume ----\n')

### Load data
siv = xr.open_dataset(os.path.join(data_dir, f'PIOMAS_Arctic_{year_min}-{year_max}_SIV_Monthly.nc'))['SIV'].sortby('time')

### Time specifications
seasons = ['Winter (DJF)', 'Spring (MAM)', 'Summer (JJA)', 'Fall (SON)']
season_months = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
colors = ['dodgerblue', 'limegreen', 'orange', 'indianred']

### Calculate seasonal sea ice volume
season_SIV, ref_season_SIV = np.empty((len(season_months), len(years))), np.empty((len(season_months), len(years)))
for i, season_month in enumerate(season_months):
    season_data = siv.sel(time=siv['time.month'].isin(season_month))
    if season_month == [12, 1, 2]:
        month, year = season_data.time.dt.month, season_data.time.dt.year
        winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
        season_data = season_data.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
        season_data = season_data.groupby('winter_year').mean(dim='time').sel(winter_year=slice(year_crop, year_max)).rename({'winter_year': 'year'})
        season_SIV[i] = season_data.values
        ref_season_SIV[i] = season_data.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')
    else:
        season_data = season_data.groupby('time.year').mean(dim='time').sel(year=slice(year_crop, year_max))
        season_SIV[i] = season_data.values
        ref_season_SIV[i] = season_data.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')
seasonal_anoms = season_SIV - ref_season_SIV

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', labelcolor='dimgray', facecolor='w')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure and adjust axes
fig, ax = plt.subplots(figsize=(10, 6))
adjust_spines(ax, ['left', 'bottom'])
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

### Horizontal lines and plot
for step, style, lw in [(2, 'solid', .5), (1, 'dotted', .5)]:
    for y in np.arange(-20, 20, step):
        plt.axhline(y, color='gray', linestyle=style, linewidth=1 if y == 0 else lw)
for j, season_month in enumerate(season_months):
    plt.plot(years, seasonal_anoms[j], linewidth=3, color=colors[j], label=f'{seasons[j]}')

### Ticks and domain
plt.xticks(np.arange(year_crop, 2040, 10), np.arange(year_crop, 2040, 10), color='k'); plt.yticks(np.arange(-10, 6.1, 2), map(str, np.arange(-10, 6.1, 2)), color='k')
plt.axis([year_crop, 2025, -10, 6])

### Title
plt.title(r'\textbf{ARCTIC SEA ICE VOLUME}', color='k', fontsize=30)

### Labels, annotation, and legend
plt.xlabel(r'\textbf{Year}', fontsize=13, color='k', labelpad=10)
plt.ylabel(r'\textbf{Sea Ice Volume Anomaly [10\textsuperscript{{3}} km\textsuperscript{{3}}]}', fontsize=13, color='k', labelpad=10)
plt.text(2025, .1, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}}}', fontsize=11, rotation='horizontal', ha='right', color='k')
l = plt.legend(shadow=False, fontsize=12, loc='upper center', bbox_to_anchor=(.5, 1.015), fancybox=True, ncol=4, frameon=False)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_PIOMAS_SIV_Anoms_Seasons_{year_crop}–{year_max}.png'), dpi=1000)
for j, season_month in enumerate(season_months):
    mask = np.isfinite(seasonal_anoms[j])
    slope, intercept, _, _, _ = linregress(years[mask], seasonal_anoms[j][mask])
    plt.plot(years, slope * years + intercept, color=colors[j], linestyle='--', linewidth=2)
    plt.text(1985.5 + 9.15 * j, 4.5, rf'{slope*10:+.2f} 10\textsuperscript{{3}} km\textsuperscript{{3}} decade\textsuperscript{{-1}}', ha='left', color=colors[j], fontsize=9)
plt.savefig(os.path.join(fig_dir, f'Reanalysis_PIOMAS_SIV_Anoms_Seasons_{year_crop}–{year_max}_linear.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')