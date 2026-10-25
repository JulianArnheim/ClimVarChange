"""
Plot the difference in Northern Hemisphere 2-m Air Temperature (T2M) between two epochs from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np
import os
import time
import xarray as xr

mpl.use('Agg')  # non-interactive backend to avoid tkinter issues

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/NH/T2M/Diff/')

### Years
year_min, year_max = 1940, 2025

### Epochs
epoch_1, epoch_2 = (1941, 1970), (1991, 2020)

print(f'\n---- Difference in 2-m Air Temperature (T2M) ----\n')

### Load data
datat = xr.open_dataset(os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_T2M_Monthly.nc'))['t2m']

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### Figure
    print(f'{text_months}')

    ### Process data
    T2M = datat.sel(time=datat['time'].dt.month.isin(mons))
    if text_months == 'DJF':
        month, year = T2M.time.dt.month, T2M.time.dt.year
        winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
        T2M = T2M.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
        T2M = T2M.groupby('winter_year').mean(dim='time').rename({'winter_year': 'year'})
    else:
        T2M = T2M.groupby('time.year').mean(dim='time')

    ### Compute difference in means
    diff_means = (T2M.sel(year=slice(epoch_2[0], epoch_2[1])).mean(dim='year') - T2M.sel(year=slice(epoch_1[0], epoch_1[1])).mean(dim='year')).sel(latitude=slice(90, 0))

    ### Spatial mean warming
    mean_diff_warming_value = diff_means.weighted(np.cos(np.deg2rad(diff_means.latitude))).mean(dim=('latitude', 'longitude')).compute().values.item()
    min_diff_warming, max_diff_warming = diff_means.min().values, diff_means.max().values
    print(f'Northern Hemisphere [Min/Mean/Max]: {min_diff_warming:+.4f} / {mean_diff_warming_value:+.4f} / {max_diff_warming:+.4f} °C')

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
    fig = plt.figure(figsize=(6.5, 3.25))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
    ax.set_xticks([])
    ax.xaxis.set_tick_params(labelsize=5); ax.yaxis.set_tick_params(labelsize=5)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontsize=5)
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='none', linewidth=.5, zorder=2); ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
    gridlines = ax.gridlines(draw_labels=True, linewidth=.5)
    gridlines.xlabel_style = gridlines.ylabel_style = {'size': 7}

    ### Plot
    cmap = plt.get_cmap('RdBu_r', lut=192)
    x, y = np.meshgrid(diff_means.longitude, diff_means.latitude)
    cf = ax.contourf(x, y, diff_means, levels=np.arange(-5, 5.1, .5), cmap=cmap, vmin=-5, vmax=5, transform=ccrs.PlateCarree(), extend='both')

    ### Colorbar
    cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.15, shrink=.8, extend='both')
    cbar.ax.tick_params(labelsize=10, color='dimgray', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
    cbar.ax.set_xticks(np.arange(-5, 6, 1)); cbar.ax.set_xticklabels(['-5', '-4', '-3', '-2', '-1', '0', '+1', '+2', '+3', '+4', '+5'])
    cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.set_xlabel(r'\textbf{Difference in 2-m Air Temperature [°C]}', fontsize=13, color='k', labelpad=10)

    ### Title
    plt.title(r'\textbf{SURFACE AIR TEMPERATURE CHANGE}', color='k', fontsize=18, y=1.18)

    ### Annotations
    plt.text(1.01, 1.3, rf'\textbf{{{"" if text_months == "ANNUAL" else f"{text_months} "}}}', fontsize=9, ha='right', va='top', transform=ax.transAxes)
    plt.text(1.05, 1.2, rf'\textbf{{{epoch_1[0]}–{epoch_1[1]}}}', fontsize=9, rotation='horizontal', ha='right', va='top', transform=ax.transAxes)
    plt.text(1, 1.1, r'\textbf{to}', fontsize=9, rotation='horizontal', ha='right', va='top', transform=ax.transAxes)
    plt.text(1.05, 1, rf'\textbf{{{epoch_2[0]}–{epoch_2[1]}}}', fontsize=9, rotation='horizontal', ha='right', va='top', transform=ax.transAxes)

    ### Save figure
    plt.subplots_adjust(top=.96, bottom=.1, left=.08, right=.92)
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_NH_T2M_Diff_{text_months}_{epoch_1[0]}–{epoch_1[1]}_{epoch_2[0]}–{epoch_2[1]}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')