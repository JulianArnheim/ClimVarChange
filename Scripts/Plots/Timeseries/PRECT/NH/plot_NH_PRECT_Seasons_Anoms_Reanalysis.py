"""
Plot Northern Hemisphere (NH) mean Precipitation (PRECT) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
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
    sign = '+' if x > 0 else '-'
    val = str(int(abs(x)))
    return f'{sign}{val}'

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/PRECT/NH/')

### Years and epoch
year_min, year_max = 1940, 2025
years = np.arange(year_min, year_max + 1)
epoch = (1981, 2010)

print(f'\n---- Northern Hemisphere Precipitation (PRECT) ----\n')

### Load data
file_precc_monthly = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_PRECC_Monthly.nc')
file_precl_monthly = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_PRECL_Monthly.nc')
ds_precc = xr.open_dataset(file_precc_monthly, chunks={'time': 30}).sortby('time')
ds_precl = xr.open_dataset(file_precl_monthly, chunks={'time': 30}).sortby('time')
PRECT = ds_precc['PRECC'] + ds_precl['PRECL']

### Time specifications
seasons = ['Winter (DJF)', 'Spring (MAM)', 'Summer (JJA)', 'Fall (SON)']
season_months = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
colors = ['dodgerblue', 'limegreen', 'orange', 'indianred']

### Process data
NH_min_lat = 0  # NH definition
NH_datap = cosine_weighted_spatial_mean(PRECT.where(PRECT['latitude'] >= NH_min_lat, drop=True)).mean(dim='longitude')

### Calculate seasonal precipitation
season_datap = np.empty((len(season_months), len(years)))
ref_season_datap = np.empty((len(season_months), len(years)))
for i, season_month in enumerate(season_months):
    season_data = NH_datap.sel(time=NH_datap['time.month'].isin(season_month))
    if season_month == [12, 1, 2]:
        month, year = season_data.time.dt.month, season_data.time.dt.year
        winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
        season_data = season_data.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
        season_data = season_data.groupby('winter_year').sum(dim='time').sel(winter_year=slice(year_min + 1, year_max)).rename({'winter_year': 'year'})
        season_datap[i] = np.nan
        season_datap[i, 1:] = season_data.values
        ref_season_datap[i] = season_data.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')
    else:
        season_data = season_data.groupby('time.year').sum(dim='time').sel(year=slice(year_min, year_max))
        season_datap[i] = season_data.values
        ref_season_datap[i] = season_data.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')
seasonal_anoms = season_datap - ref_season_datap

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

### Horizontal lines
for step, style, lw in [(10, 'solid', .5), (5, 'dotted', .5)]:
    for y in np.arange(-100, 100, step):
        plt.axhline(y, color='gray', linestyle=style, linewidth=1 if y == 0 else lw)

### Plot
for j, season_month in enumerate(season_months):
    plt.plot(years, seasonal_anoms[j], linewidth=3, color=colors[j], label=f'{seasons[j]}')

### Ticks and domain
plt.xticks(np.arange(1940, 2040, 10), np.arange(1940, 2040, 10), color='k')
plt.yticks(np.arange(-30, 35, 5), map(str, np.arange(-30, 35, 5)), color='k')
plt.xlim([1940, 2025]); plt.ylim([-25, 25])

### Title, labels, annotation, and legend
plt.title(r'\textbf{NORTHERN HEMISPHERE MEAN PRECIPITATION}', color='k', fontsize=25)
plt.xlabel(r'\textbf{Year}', fontsize=13, color='k', labelpad=10)
plt.ylabel(r'\textbf{Precipitation Anomaly [mm season\textsuperscript{-1}]}', fontsize=13, color='k', labelpad=10)
plt.text(2025, -24.5, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}}}', fontsize=11, rotation='horizontal', ha='right', color='k')
l = plt.legend(shadow=False, fontsize=12, loc='upper center', bbox_to_anchor=(.5, 1.025), fancybox=True, ncol=4, frameon=False)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_NH_PRECT_Anoms_Seasons_{year_min}–{year_max}.png'), dpi=1000)
for j, season_month in enumerate(season_months):
    mask = np.isfinite(seasonal_anoms[j])
    slope, intercept, _, _, _ = linregress(years[mask], seasonal_anoms[j][mask])
    plt.plot(years, slope * years + intercept, color=colors[j], linestyle='--', linewidth=2)
    plt.text(1956 + 18 * j, 21.5, rf'{slope*10:+.2f} mm\textsuperscript{{-1}} season\textsuperscript{{-1}} decade\textsuperscript{{-1}}', ha='center', color=colors[j], fontsize=9)
plt.savefig(os.path.join(fig_dir, f'Reanalysis_NH_PRECT_Anoms_Seasons_{year_min}–{year_max}_linear.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')