"""
Plot Northern Hemisphere (NH) monthly Sea Ice Extent (SIE) anomalies, the area of grid cells with at least 15% Sea Ice Concentration (ICEFRAC), from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
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
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks_position('bottom')

def custom_formatter(x, _):
    if abs(x) < .01:
        return '0'
    return f'{x:+.0f}'

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

### Time specifications
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
colors = ['#B0C4DE', '#9ACD32', '#FFC0CB', '#D2691E', '#00FFFF', '#FFA500',
          '#FF69B4', '#32CD32', '#FFD700', '#800080', '#FF0000', '#7B68EE']

print(f'\n---- Arctic Sea Ice Extent ----\n')

### Load data
filename = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_ICEFRAC_Monthly.nc')
datac = xr.open_dataset(filename)['siconc']

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

### Monthly sea ice extent
ice_extent_obl_mkm2 = cell_area_obl.where(ICEFRAC >= 15).sum(dim=['latitude', 'longitude']) / 1e12

### Annual mean sea ice extent anomalies
baseline_extent = ice_extent_obl_mkm2.sel(time=slice(f'{epoch[0]}-01-01', f'{epoch[1]}-12-31')).mean(dim='time')
annual_mean_anom = ice_extent_obl_mkm2.groupby('time.year').mean(dim='time') - baseline_extent

### Monthly sea ice extent anomalies
SIE_by_month = {}
for month, group in ice_extent_obl_mkm2.groupby('time.month'):
    monthly_refmean = group.sel(time=slice(f'{epoch[0]}-01-01', f'{epoch[1]}-12-31')).mean(dim='time')
    SIE_by_month[month] = group - monthly_refmean

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
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

### Ticks
x_ticks = np.arange(year_crop, 2026, 5)
y_ticks = np.arange(-4, 3.1, 1)
plt.xticks(x_ticks, x_ticks, color='k')
plt.yticks(y_ticks, map(str, y_ticks), color='k')

### Domain
plt.xlim([year_crop, 2025])
plt.ylim([-4, 2])

### Horizontal lines
for step, style, lw in [(1, 'solid', .5), (.5, 'dotted', .5)]:
    for y in np.arange(-10, 10, step):
        plt.axhline(y, color='gray', linestyle=style, linewidth=1 if y == 0 else lw)

### Plot
for month, anom in SIE_by_month.items():
    plt.plot(anom['time.year'], anom, color=colors[month - 1], label=months[month - 1])
plt.plot(annual_mean_anom['year'], annual_mean_anom.values, color='k', linewidth=3)

### Title
plt.title(r'\textbf{ARCTIC SEA ICE EXTENT}', color='k', fontsize=30)

### Labels
plt.xlabel(r'\textbf{Year}', fontsize=13, color='k', labelpad=10)
plt.ylabel(r'\textbf{Sea Ice Extent Anomaly [million km\textsuperscript{{2}}]}', fontsize=13, color='k', labelpad=10)

### Annotation
plt.text(2025, -3.95, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}}}', fontsize=11, ha='right', color='k')

### Legend
l = plt.legend(shadow=False, fontsize=11, loc='upper center', bbox_to_anchor=(.5, 1.02), fancybox=True, ncol=6, frameon=False, labelspacing=.15)
for text in l.get_texts():
    text.set_color('k')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_ICEFRAC_SIE_Anoms_Months_{year_min}–{year_max}.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')