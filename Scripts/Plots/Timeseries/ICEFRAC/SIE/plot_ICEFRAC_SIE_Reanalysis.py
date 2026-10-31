"""
Plot Northern Hemisphere (NH) mean Sea Ice Extent (SIE), the area of grid cells with at least 15% Sea Ice Concentration (ICEFRAC), from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
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
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/ICEFRAC/SIE/')

### Years and epoch
year_crop = 1980
year_min, year_max = 1940, 2025
epoch = (1981, 2010)

print(f'\n---- Arctic Sea Ice Extent ----\n')

### Load data
filename = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_ICEFRAC_Monthly.nc')
datac = xr.open_dataset(filename)['siconc']

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', labelcolor='dimgray', facecolor='w')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Process data
ICEFRAC = datac.where(datac.latitude >= 30, drop=True) * 100

### Grid cell area for an oblate spheroid Earth
R_eq, R_polar = 6378137.0, 6356752.3  # equatorial and polar radii [m]
e2 = 1 - (R_polar**2 / R_eq**2)  # eccentricity^2
lat_rad = np.deg2rad(ICEFRAC.latitude.values)
lon_rad = np.deg2rad(ICEFRAC.longitude.values)
dlat, dlon = np.abs(np.diff(lat_rad).mean()), np.abs(np.diff(lon_rad).mean())
Rm = R_eq * (1 - e2) / (1 - e2 * np.sin(lat_rad) ** 2) ** 1.5
Rp = R_eq / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)
cell_area_obl = Rm * Rp * np.cos(lat_rad) * dlat * dlon
cell_area_obl = np.broadcast_to(cell_area_obl[:, np.newaxis], (len(lat_rad), len(lon_rad)))
cell_area_obl = xr.DataArray(cell_area_obl, coords=[ICEFRAC.latitude, ICEFRAC.longitude], dims=['latitude', 'longitude'])

### Calculate sea ice extent means and anomalies
ice_extent_obl_mkm2 = cell_area_obl.where(ICEFRAC >= 15).sum(dim=['latitude', 'longitude']) / 1e12
ice_extent_obl_mkm2 = ice_extent_obl_mkm2.groupby('time.year').mean(dim='time')
baseline_extent = ice_extent_obl_mkm2.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')
ice_extent_anom = ice_extent_obl_mkm2 - baseline_extent
years, z_anom = ice_extent_anom.year.values, ice_extent_anom.values

### LOWESS smoothing
years, z_anom = ice_extent_anom['year'].values, ice_extent_anom.values
lowess_frac = 30 / (years[-1] - years[0])  # fraction ~ 30-year window
z_lowess = lowess(endog=z_anom, exog=years, frac=lowess_frac, return_sorted=False)

### Create figure
fig, ax = plt.subplots(figsize=(10, 6))
adjust_spines(ax, ['left', 'bottom'])
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

### Ticks and domain
plt.xticks(np.arange(1940, 2040, 10), np.arange(1940, 2040, 10), color='k'); plt.yticks(np.arange(-2, 1.1, 1), map(str, np.arange(-2, 1.1, 1)), color='k')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
ax.set_xlim([year_crop, year_max]); ax.set_ylim([-2, 1])

### Horizontal lines
for i in np.arange(-10, 11, .5):
    plt.axhline(i, color='gray', linewidth=1 if i == 0 else .5, linestyle='-' if i % 1 == 0 else 'dotted')

### Plot
plt.plot(years, z_anom, color='k', linewidth=3, label=f'Annual Mean')
plt.plot(years, z_lowess, color='deeppink', linewidth=3, label='30-Year LOWESS')

### Title, labels, annotation, and legend
plt.title(rf'\textbf{{ARCTIC SEA ICE EXTENT}}', color='k', fontsize=30)
plt.xlabel(r'\textbf{Year}', fontsize=16, color='k', labelpad=10); plt.ylabel(r'\textbf{Sea Ice Extent Anomaly [million km\textsuperscript{{2}}]}', fontsize=16, color='k', labelpad=10)
plt.text(2020, .03, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}, {float(baseline_extent):.2f} million km\textsuperscript{{2}}}}', fontsize=11, ha='right', color='k')
plt.legend(shadow=False, fontsize=13, loc='upper center', bbox_to_anchor=(.5, 1), fancybox=True, ncol=2, frameon=False)

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_ICEFRAC_SIE_Anoms_{year_crop}–{year_max}.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')