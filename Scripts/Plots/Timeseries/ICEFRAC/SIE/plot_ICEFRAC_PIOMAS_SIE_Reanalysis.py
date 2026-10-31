"""
Plot Northern Hemisphere (NH) mean Sea Ice Extent (SIE), the area of grid cells with at least 15% Sea Ice Concentration (ICEFRAC), for ERA5 and PIOMAS reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Data : https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/
    Author : Julian B Arnheim
"""

### Import modules
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

def annual_extent_anomaly(ICEFRAC, epoch):
    R_eq, R_polar = 6378137.0, 6356752.3  # equatorial and polar radii [m]
    e2 = 1 - (R_polar**2 / R_eq**2)  # eccentricity^2
    lat_rad, lon_rad = np.deg2rad(ICEFRAC.latitude.values), np.deg2rad(ICEFRAC.longitude.values)
    dlat, dlon = np.abs(np.diff(lat_rad).mean()), np.abs(np.diff(lon_rad).mean())
    Rm = R_eq * (1 - e2) / (1 - e2 * np.sin(lat_rad) ** 2) ** 1.5
    Rp = R_eq / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)
    cell_area_obl = np.broadcast_to((Rm * Rp * np.cos(lat_rad) * dlat * dlon)[:, np.newaxis], (len(lat_rad), len(lon_rad)))
    cell_area_obl = xr.DataArray(cell_area_obl, coords=[ICEFRAC.latitude, ICEFRAC.longitude], dims=['latitude', 'longitude'])
    ice_extent_obl_mkm2 = cell_area_obl.where(ICEFRAC >= 15).sum(dim=['latitude', 'longitude']) / 1e12
    ice_extent_obl_mkm2 = ice_extent_obl_mkm2.groupby('time.year').mean(dim='time')
    baseline_extent = ice_extent_obl_mkm2.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')
    return ice_extent_obl_mkm2 - baseline_extent, float(baseline_extent)

def custom_formatter(x, _):
    if abs(x) < .01:
        return '0'
    sign, val = '+' if x > 0 else '-', f'{abs(x):.0f}'.lstrip('0')
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
piomas_min, piomas_max = 1979, 2025
epoch = (1981, 2010)

print(f'\n---- Arctic Sea Ice Extent: ERA5 and PIOMAS ----\n')

### Load data
era5 = xr.open_dataset(os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_ICEFRAC_Monthly.nc'))['siconc']
piomas = xr.open_dataset(os.path.join(data_dir, f'PIOMAS_Arctic_{piomas_min}-{piomas_max}_SIC_Monthly.nc'))['SIC']

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', labelcolor='dimgray', facecolor='w')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Process data
ICEFRAC_era5 = era5.where(era5.latitude >= 30, drop=True) * 100
ICEFRAC_piomas = piomas.where(piomas.latitude >= 30, drop=True) * 100

### Calculate sea ice extent anomalies for each dataset
era5_anom, era5_baseline = annual_extent_anomaly(ICEFRAC_era5, epoch)
piomas_anom, piomas_baseline = annual_extent_anomaly(ICEFRAC_piomas, epoch)

### Create figure
fig, ax = plt.subplots(figsize=(10, 6))
adjust_spines(ax, ['left', 'bottom'])
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

### Ticks and domain
plt.xticks(np.arange(1940, 2040, 10), np.arange(1940, 2040, 10), color='k')
plt.yticks(np.arange(-2, 1.1, 1), map(str, np.arange(-2, 1.1, 1)), color='k')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))
ax.axis([year_crop, year_max, -2, 1])

### Horizontal lines
for i in np.arange(-10, 11, .5):
    plt.axhline(i, color='gray', linewidth=1 if i == 0 else .5, linestyle='-' if i % 1 == 0 else 'dotted')

### Plot
plt.plot(era5_anom['year'].values, era5_anom.values, color='k', linewidth=3, linestyle='-', label='ERA5')
plt.plot(piomas_anom['year'].values, piomas_anom.values, color='k', linewidth=2, linestyle='--', label='PIOMAS')

### Title
plt.title(rf'\textbf{{ARCTIC SEA ICE EXTENT}}', color='k', fontsize=30)

### Label
plt.xlabel(r'\textbf{Year}', fontsize=16, color='k', labelpad=10); plt.ylabel(r'\textbf{Sea Ice Extent Anomaly [million km\textsuperscript{{2}}]}', fontsize=16, color='k', labelpad=10)

### Annotation
plt.text(2025, -1.97, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}, ERA5 {era5_baseline:.2f}, PIOMAS {piomas_baseline:.2f} million km\textsuperscript{{2}}}}', fontsize=11, ha='right', color='k')

### Legend
plt.legend(shadow=False, fontsize=13, loc='upper center', bbox_to_anchor=(.5, 1), fancybox=True, ncol=2, frameon=False, handlelength=3)

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_ICEFRAC_PIOMAS_SIE_Anoms_{year_crop}–{year_max}.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')