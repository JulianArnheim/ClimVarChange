"""
Plot North Atlantic Oscillation (NAO) timeseries from NOAA CPC

Notes
-----
    Data : https://ftp.cpc.ncep.noaa.gov/wd52dg/data/indices/nao_index.tim
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib.ticker import MultipleLocator
from shapely.geometry import LineString, Point
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import os
import statsmodels.api as sm
import time
import xarray as xr

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')

def form_sign(x, _):
    if abs(x) < 1e-10:
        return '0'
    return f'{x:+.1f}'.replace('+0.', '+.').replace('-0.', '-.').replace('.0', '')

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/NAO/')

### Years
year_min, year_max = 1950, 2025

### Canonical NAO dipole centers from Wallace & Gutzler (1981) and Hurrell (1995)
NAO_pos_lat, NAO_pos_lon = 35, -25  # Azores High: ~35°N, 25°W
NAO_neg_lat, NAO_neg_lon = 65, -22  # Icelandic Low: ~65°N, 22°W

print(f'\n---- North Atlantic Oscillation (NAO) Timeseries ----\n')

### Month options
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Load NAO data
NAOs = xr.open_dataset(os.path.join(data_dir, f'CPC_NAO_EA_1950-{year_max}_Monthly.nc'))

### Loop through each season
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### For title only, make ANNUAL blank
    season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '

    ### Figure
    print(f'{text_months}')

    ### Process data
    NAO_data = NAOs.sel(time=slice(f'{year_min}-01-01', f'{year_max}-12-31')).where(NAOs.time.dt.month.isin(mons), drop=True)
    if text_months == 'DJF':
        month, year = NAO_data.time.dt.month, NAO_data.time.dt.year
        winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
        NAO_annual = NAO_data.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
        NAO_annual = NAO_annual.groupby('winter_year').mean(dim='time').sel(winter_year=slice(year_min + 1, year_max)).rename({'winter_year': 'year'})
    else:
        NAO_annual = NAO_data.groupby('time.year').mean(dim='time')

    ### Figure design
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='w')
    plt.rc('axes', edgecolor='dimgray')
    plt.rc('xtick', labelsize=12, labelcolor='dimgray')
    plt.rc('ytick', labelsize=12, labelcolor='dimgray')
    plt.rc('axes', labelcolor='dimgray')
    plt.rc('axes', facecolor='w')

    ### Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    ### Adjust axes
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

    ### Ticks and domain
    x_years, xtick_vals = NAO_annual.year.values, np.arange(1940, 2031, 5)
    xtick_labels = [str(y) for y in xtick_vals]
    plt.xticks(xtick_vals, xtick_labels, fontsize=12, family='Avant Garde', color='k')
    ax.set_xlim([1950, 2025])

    ### Horizontal lines
    for y in np.arange(-5, 5, .5):
        ax.axhline(y, color='gray', linestyle='dashed', linewidth=.5, zorder=1)
    for y in np.arange(-2, 3, 1):
        ax.axhline(y, color='dimgray' if y == 0 else 'gray', linewidth=1.5 if y == 0 else .5, zorder=1)

    ### Plot NAO and LOWESS smoothed timeseries
    ax.plot(x_years, NAO_annual['NAO'], color='g', label='NAO', linewidth=1.5)
    ax.set_ylim([-2.5, 2.5])
    ax.yaxis.set_major_locator(MultipleLocator(.5)); ax.yaxis.set_major_formatter(plt.FuncFormatter(form_sign))
    ax.tick_params(axis='y', length=5.5, width=2, which='major', colors='dimgray', labelcolor='k', labelsize=12)
    ax.set_ylabel(r'\textbf{NAO}', fontsize=16, color='k', labelpad=12)
    lowess_num = 30  # years
    lowess_smoothed_NAO = sm.nonparametric.lowess(NAO_annual['NAO'].values, x_years, frac=lowess_num / len(x_years))
    ax.plot(lowess_smoothed_NAO[:, 0], lowess_smoothed_NAO[:, 1], color='g', linewidth=3)

    ### Title and annotations
    plt.title(rf'\textbf{{{season_prefix} North Atlantic Oscillation (NAO)}}', color='k', fontsize=22, y=1)
    ax.text(.5, .96, rf'\textbf{{{lowess_num}-Year LOWESS Smoothing}}', transform=ax.transAxes, ha='center', va='top', color='g', fontsize=13)

    plt.tight_layout()

    ### Inset region with NAO dipole centers
    inset_ax = fig.add_axes([.07, .63, .28, .28], projection=ccrs.NearsidePerspective(central_longitude=-37, central_latitude=63, satellite_height=50000000), zorder=3)
    inset_ax.set_global()
    inset_ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor='gray', zorder=4); inset_ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='w', zorder=5)
    inset_ax.coastlines(resolution='50m', linewidth=.5, zorder=6)
    for lat in [-60, -30, 0, 30, 60]:
        line = LineString([(lon, lat) for lon in np.linspace(-180, 180, 361)])
        inset_ax.add_geometries([line], crs=ccrs.PlateCarree(), edgecolor='k', facecolor='none', linestyle='--', linewidth=.5, alpha=1, zorder=8)
    N_pole, S_pole = Point(0, 90), Point(0, -90)
    inset_ax.scatter([N_pole.x, S_pole.x], [N_pole.y, S_pole.y], transform=ccrs.PlateCarree(), color='k', s=1, alpha=1, zorder=9)
    inset_ax.scatter([NAO_pos_lon, NAO_neg_lon], [NAO_pos_lat, NAO_neg_lat], transform=ccrs.PlateCarree(), marker='*', s=150, color='gold', edgecolor='k', linewidth=.6, zorder=10)

    ### Save figure
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_NAO_{text_months}_{year_min}–{year_max}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')