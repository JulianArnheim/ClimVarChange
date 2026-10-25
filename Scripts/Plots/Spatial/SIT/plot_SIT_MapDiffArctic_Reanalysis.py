"""
Plot difference in Arctic Sea Ice Thickness (SIT) between two epochs from the PIOMAS ice-ocean model

Notes
-----
    Data : https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/model_grid
    Author : Julian B Arnheim
"""

### Import modules
from cartopy.util import add_cyclic_point
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs, cartopy.feature as cfeature
import cmocean
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import xarray as xr

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Arctic/SIT/Diff/')

### Years and epochs
year_min, year_max, epoch_1, epoch_2 = 1979, 2025, (1980, 1999), (2006, 2025)

print(f'\n---- Difference in Sea Ice Thickness (SIT) ----\n')

### Load data
datathick = xr.open_dataset(os.path.join(data_dir, f'PIOMAS_Arctic_{year_min}-{year_max}_SIT_Monthly.nc'))['SIT']

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### Figure
    print(f'{text_months}')

    ### Process data
    SIT = datathick.sel(time=datathick['time'].dt.month.isin(mons))
    if text_months == 'DJF':
        month, year = SIT.time.dt.month, SIT.time.dt.year
        winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
        SIT = SIT.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
        SIT = SIT.groupby('winter_year').mean(dim='time').rename({'winter_year': 'year'})
    else:
        SIT = SIT.groupby('time.year').mean(dim='time')
    SIT = SIT.where(SIT.latitude >= 0, drop=True)

    ### Compute difference in means
    diff_means = SIT.sel(year=slice(epoch_2[0], epoch_2[1])).mean(dim='year') - SIT.sel(year=slice(epoch_1[0], epoch_1[1])).mean(dim='year')

    ### Arctic sea ice statistics
    diff_means_min, diff_means_max = diff_means.min().item(), diff_means.max().item()
    print(f'Arctic sea ice change [Min/Max]: {diff_means_min:+.2f} / {diff_means_max:+.2f} m')

    ### Spatial mean thickness change
    ice_mask = SIT.sel(year=slice(epoch_1[0], epoch_2[1])).mean(dim='year') >= .15
    masked = diff_means.where(ice_mask)
    mean_val = masked.weighted(np.cos(np.deg2rad(masked.latitude))).mean(dim=('latitude', 'longitude')).compute().item()
    min_thickness, max_thickness = diff_means.min().compute().item(), diff_means.max().compute().item()
    print(f'Arctic [Min/Mean/Max]: {min_thickness:.4f} / {mean_val:.4f} / {max_thickness:.4f} m')

    ### Figure design
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='w')
    plt.rc('axes', edgecolor='k')
    plt.rc('xtick', color='dimgray')
    plt.rc('ytick', color='dimgray')
    plt.rc('axes', labelcolor='dimgray')
    plt.rc('axes', facecolor='w')

    ### Create figure
    fig = plt.figure(figsize=(5.5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=.5, resolution='50m', zorder=4)
    ax.set_xticks([])
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontsize=5)

    ### Fill missing values at the pole with the mean of the nearest latitude
    lat_max = diff_means.latitude.max()
    lat_below = diff_means.latitude.where(diff_means.latitude < lat_max, drop=True).max()
    diff_means.loc[dict(latitude=lat_max)] = diff_means.sel(latitude=lat_below).mean(dim='longitude')

    ### Plot
    data_cyclic, lon_cyclic = add_cyclic_point(diff_means.values, coord=diff_means.longitude)
    x_wrap, y_wrap = np.meshgrid(lon_cyclic, diff_means.latitude.values)
    levels = np.arange(-2, 2.1, .25)
    cf = ax.contourf(x_wrap, y_wrap, data_cyclic, levels=levels, cmap=cmocean.cm.balance_r, vmin=-2, vmax=2, extend='both', transform=ccrs.PlateCarree())

    ### Features
    ax.add_feature(cfeature.LAND, edgecolor='k', facecolor='lightgray', linewidth=.5, zorder=1)
    ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
    ax.coastlines(linewidth=.5, resolution='50m', zorder=3)

    ### Gridlines
    for lon in np.arange(-180, 181, 60):
        ax.plot([lon, lon], [0, 90], transform=ccrs.PlateCarree(), color='gray', linewidth=.5, zorder=5)
    for lat in np.arange(0, 91, 10):
        ax.plot(np.linspace(-180, 180, 361), np.full(361, lat), transform=ccrs.PlateCarree(), color='gray', linewidth=.5, zorder=5)

    ### Colorbar
    cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.06, shrink=.8, extend='both')
    cbar.ax.tick_params(labelsize=10, colors='k', labelright=True, labelleft=False, right=True, left=False)
    cbar.ax.set_xticks(np.linspace(-2, 2, 5))
    cbar.ax.set_xticklabels(['-2', '-1', '0', '+1', '+2'])
    cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.set_xlabel(r'\textbf{Difference in Sea Ice Thickness [m]}', fontsize=13, color='k', labelpad=8)
    cbar.outline.set_edgecolor('k')
    cbar.outline.set_linewidth(.8)

    ### Title
    plt.title(r'\textbf{SEA ICE CHANGE}', color='k', fontsize=20, y=1.05)

    ### Annotations
    season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
    plt.text(1, 1.04, rf'\textbf{{{season_prefix}{epoch_1[0]}–{epoch_1[1]} to {epoch_2[0]}–{epoch_2[1]}}}', fontsize=10, rotation='horizontal',
             ha='right', va='top', transform=ax.transAxes)
    plt.text(.02, .98, rf'\textbf{{{mean_val:.2f} m}}', fontsize=9, rotation='horizontal', ha='left', va='top', transform=ax.transAxes,
             bbox=dict(facecolor='w', edgecolor='k', boxstyle='round,pad=0.2'), zorder=6)

    ### Ensure outline is on top
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none', edgecolor='k', linewidth=1.2, zorder=10))

    ### Save figure
    plt.subplots_adjust(top=.88, bottom=.02, left=.02, right=.98)
    plt.savefig(os.path.join(fig_dir, f'PIOMAS_Arctic_SIT_Diff_{text_months}_{epoch_1[0]}–{epoch_1[1]}_{epoch_2[0]}–{epoch_2[1]}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')