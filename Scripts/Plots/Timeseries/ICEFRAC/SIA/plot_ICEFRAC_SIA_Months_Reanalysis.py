"""
Plot Northern Hemisphere (NH) monthly Sea Ice Area (SIA), derived from Sea Ice Concentration (ICEFRAC), from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib.colors import LinearSegmentedColormap
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
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/ICEFRAC/SIA/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Arctic Sea Ice Area ----\n')

### Time specifications
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
month_positions = [1, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365]

### Load data
filename = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_ICEFRAC_Monthly.nc')
datac = xr.open_dataset(filename)['siconc']

### Process data
ICEFRAC = datac.sortby('time').where(datac['latitude'] >= 30, drop=True) * 100

### Arctic sea ice area for an oblate spheroid Earth
R_eq, R_polar = 6378137.0, 6356752.3  # equatorial and polar radii [m]
e2 = 1 - (R_polar**2 / R_eq**2)  # eccentricity^2
lat_rad = np.deg2rad(ICEFRAC.latitude.values)
lon_rad = np.deg2rad(ICEFRAC.longitude.values)
dlat = np.abs(np.diff(lat_rad).mean())
dlon = np.abs(np.diff(lon_rad).mean())
Rm = R_eq * (1 - e2) / (1 - e2 * np.sin(lat_rad) ** 2) ** 1.5
Rp = R_eq / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)
cell_area_obl = Rm * Rp * np.cos(lat_rad) * dlat * dlon
cell_area_obl = np.broadcast_to(cell_area_obl[:, np.newaxis], (len(lat_rad), len(lon_rad)))
cell_area_obl = xr.DataArray(cell_area_obl, coords=[ICEFRAC.latitude, ICEFRAC.longitude], dims=['latitude', 'longitude'])

### Calculate monthly sea ice area
ice_area_obl_mkm2 = (cell_area_obl * ICEFRAC / 100).where(ICEFRAC).sum(dim=['latitude', 'longitude']) / 1e12

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

### Ticks
yticks = np.arange(0, 21, 2)
ax.set_xticks(month_positions, month_labels, fontsize=12, color='k', rotation=0)
ax.set_yticks(yticks, map(str, yticks), color='k')

### Domain
ax.set_xlim(1, 365)
ax.set_ylim([2, 16])

### Horizontal lines
for step, style, lw in [(2, 'solid', .5), (1, 'dotted', .5)]:
    for y in np.arange(0, 21, step):
        plt.axhline(y, color='gray', linestyle=style, linewidth=lw)

### Plot
colors = ['#ffeda0', '#fdd49e', '#fdbb84', '#fc8d59',
          '#ef6548', '#d7301f', '#b30000', '#7f0000', '#000000']
years_sorted = np.unique(ice_area_obl_mkm2['time.year'].values)
num_years = len(years_sorted)
blue_cmap = LinearSegmentedColormap.from_list('blue_cmap', colors, num_years)
for idx, year in enumerate(years_sorted):

    ### Start with months Jan–Dec
    group = ice_area_obl_mkm2.sel(time=str(year))
    full_year = group.values.copy()
    months = month_positions[:-1]

    ### Fetch previous December if it exists
    if idx > 0:
        prev_dec = ice_area_obl_mkm2.sel(time=f'{years_sorted[idx - 1]}-12').item()
        full_year = np.concatenate([[prev_dec], full_year])
        months = [month_positions[0] - 31] + months

    ### Fetch next January if it exists
    if idx < len(years_sorted) - 1:
        next_jan = ice_area_obl_mkm2.sel(time=f'{years_sorted[idx + 1]}-01').item()
        full_year = np.concatenate([full_year, [next_jan]])
        months = months + [month_positions[-1] + 31]

    plt.plot(months, full_year, color=blue_cmap(idx / (num_years - 1)))

### Title
plt.title(r'\textbf{ARCTIC SEA ICE AREA}', color='k', fontsize=30)

### Labels
plt.xlabel(r'\textbf{Month}', fontsize=13, color='k', labelpad=10)
plt.ylabel(r'\textbf{Sea Ice Area [million km\textsuperscript{{2}}]}', fontsize=13, color='k', labelpad=10)

### Legend
decades = [year for year in years_sorted if year % 10 == 0]
l_handles = [plt.Line2D([0], [0], color=blue_cmap(i / (len(decades) - 1)), lw=2) for i, _ in enumerate(decades)]
l = plt.legend(handles=l_handles, labels=[f'{year}s' for year in decades], shadow=False, fontsize=4, loc='upper left',
               bbox_to_anchor=(.55, .99), fancybox=True, ncol=3, frameon=False, prop={'weight': 'bold'}, labelspacing=1.4)

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_ICEFRAC_SIA_{year_min}–{year_max}.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')