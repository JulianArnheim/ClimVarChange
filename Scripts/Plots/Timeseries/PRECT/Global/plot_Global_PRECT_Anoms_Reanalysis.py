"""
Plot global mean Precipitation (PRECT) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from statsmodels.nonparametric.smoothers_lowess import lowess
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
    ax.xaxis.set_ticks_position('bottom')

def cosine_weighted_spatial_mean(data, coord=None):
    coord = coord or ('lat' if 'lat' in data.coords else 'latitude' if 'latitude' in data.coords else None)
    if not coord:
        raise ValueError("No 'lat' or 'latitude' coordinate found in the data array.")
    if 'lat' in data.coords and 'latitude' in data.coords:
        raise ValueError("Both 'lat' and 'latitude' coordinates exist.")
    weights = np.cos(np.deg2rad(data[coord]))
    return (data * weights).sum(dim=coord) / (data.notnull() * weights).sum(dim=coord)

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/PRECT/Global/')

### Years and epoch
year_min, year_max = 1940, 2025
epoch = (1981, 2010)

print(f'\n---- Precipitation (PRECT) ----\n')

### Load data (PRECC + PRECL → PRECT)
file_precc_monthly = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_PRECC_Monthly.nc')
file_precl_monthly = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_PRECL_Monthly.nc')
file_prect_annual = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_PRECT_Annual.nc')
if os.path.exists(file_prect_annual):
    print(f'Annual file exists, loading...\n')
    datap_annual = xr.open_dataset(file_prect_annual, chunks={'year': 10})
else:
    print(f'Computing annual from monthly PRECC + PRECL...')
    ds_precc = xr.open_dataset(file_precc_monthly, chunks={'time': 30}).sortby('time')
    ds_precl = xr.open_dataset(file_precl_monthly, chunks={'time': 30}).sortby('time')
    PRECT = ds_precc['PRECC'] + ds_precl['PRECL']
    datap_annual = xr.Dataset({'PRECT': PRECT}).groupby('time.year').sum(dim='time')
    datap_annual.to_netcdf(file_prect_annual)
    print(f'Saved {file_prect_annual}, the annual PRECT means\n')

### Calculate annual means and anomalies
annual_mean_datap = cosine_weighted_spatial_mean(datap_annual.mean(dim='longitude'))
refmean_datap = annual_mean_datap.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')
mean_datap_anom = annual_mean_datap - refmean_datap

### LOWESS smoothing
years = mean_datap_anom['year'].values
z_anom = mean_datap_anom['PRECT'].values
lowess_frac = 30 / (years[-1] - years[0])  # fraction ~ 30-year window
z_lowess = lowess(endog=z_anom, exog=years, frac=lowess_frac, return_sorted=False)

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', labelcolor='dimgray', facecolor='w')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure
fig, ax = plt.subplots(figsize=(10, 6))

### Adjust axes
adjust_spines(ax, ['left', 'bottom'])
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

### Ticks
xtick_vals, ytick_vals = np.arange(1940, 2031, 5), np.arange(-40, 41, 10)
plt.xticks(xtick_vals, [str(y) for y in xtick_vals], fontsize=12, family='Avant Garde', color='k')
plt.yticks(ytick_vals, [f'+{y:d}' if y > 0 else f'{y:d}' for y in ytick_vals], fontsize=12, family='Avant Garde', color='k')

### Domain
plt.xlim([1940, 2025]); plt.ylim([-40, 40])

### Horizontal lines
for i in np.arange(-40, 41, 10):
    plt.axhline(i, color='gray', linewidth=1 if i == 0 else .5, linestyle='-' if i == 0 else 'dotted')

### Plot
plt.plot(years, z_anom, color='k', linewidth=3, label='Annual Mean')
plt.plot(years, z_lowess, color='deeppink', linewidth=3, label='30-Year LOWESS')

### Title, labels, annotation, and legend
plt.title(r'\textbf{GLOBAL MEAN PRECIPITATION}', color='k', fontsize=30)
plt.xlabel(r'\textbf{Year}', fontsize=16, color='k', labelpad=10)
plt.ylabel(r'\textbf{Precipitation [mm yr\textsuperscript{-1}]}', fontsize=16, color='k', labelpad=10)
plt.text(2025, -39, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}, at $\sim${refmean_datap["PRECT"].values:,.0f} mm yr\textsuperscript{{-1}}}}', fontsize=11, ha='right', color='k')
l = plt.legend(shadow=False, fontsize=13, loc='upper center', bbox_to_anchor=(.5, 1), fancybox=True, ncol=2, frameon=False)

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_Global_PRECT_Anoms_{year_min}–{year_max}.png'), dpi=1000)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')