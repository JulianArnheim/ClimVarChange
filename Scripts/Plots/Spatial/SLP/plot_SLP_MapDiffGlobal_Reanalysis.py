"""
Plot the difference in Sea Level Pressure (SLP) between two epochs from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import cartopy.crs as ccrs, cartopy.feature as cfeature
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
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Global/SLP/Diff/')

### Years and epochs
year_min, year_max = 1940, 2025
epoch_1, epoch_2 = (1941, 1970), (1991, 2020)

print(f'\n---- Difference in Sea Level Pressure (SLP) ----\n')

### Load data
filename = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_SLP_Monthly.nc')
datap = xr.open_dataset(filename)['SLP']

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### Figure
    print(f'{text_months}')

    ### Process data
    SLP = datap.sel(time=datap['time'].dt.month.isin(mons))
    if text_months == 'DJF':
        month, year = SLP.time.dt.month, SLP.time.dt.year
        winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
        SLP = SLP.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
        SLP = SLP.groupby('winter_year').mean(dim='time').rename({'winter_year': 'year'})
    else:
        SLP = SLP.groupby('time.year').mean(dim='time')

    ### Compute difference in means
    diff_means = SLP.sel(year=slice(epoch_2[0], epoch_2[1])).mean(dim='year') - SLP.sel(year=slice(epoch_1[0], epoch_1[1])).mean(dim='year')

    ### Spatial mean sea level pressure change
    weighted_diff = diff_means.weighted(np.cos(np.deg2rad(diff_means.latitude)))
    mean_diff_warming = weighted_diff.mean(dim=('latitude', 'longitude'))
    mean_diff_warming_value = mean_diff_warming.compute().values.item()
    min_diff_warming = diff_means.min().values
    max_diff_warming = diff_means.max().values
    print(f'Global [Min/Mean/Max]: {min_diff_warming:+.4f} / {mean_diff_warming_value:+.4f} / {max_diff_warming:+.4f} hPa')

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
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_xticks([])
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontsize=5)
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='none', linewidth=.5, zorder=2)
    ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
    gridlines = ax.gridlines(draw_labels=True, linewidth=.5, zorder=4)
    gridlines.xlabel_style = gridlines.ylabel_style = {'size': 7}

    ### Plot
    x, y = np.meshgrid(datap.longitude, datap.latitude)
    cf = ax.contourf(x, y, diff_means, levels=np.arange(-5, 5.1, .5), cmap=plt.get_cmap('RdBu_r', lut=192), vmin=-5, vmax=5, transform=ccrs.PlateCarree(), extend='both')

    ### Colorbar
    cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.1, shrink=.8, extend='both')
    cbar.ax.tick_params(labelsize=10, color='dimgray', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
    cbar.ax.set_xticks(np.arange(-5, 6, 1)); cbar.ax.set_xticklabels(['-5', '-4', '-3', '-2', '-1', '0', '+1', '+2', '+3', '+4', '+5'])
    cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.set_xlabel(r'\textbf{Difference in Sea Level Pressure [hPa]}', fontsize=13, color='k', labelpad=10)

    ### Title
    plt.title(rf'\textbf{{SEA LEVEL PRESSURE CHANGE}}', color='k', fontsize=18, y=1.12)

    ### Annotations
    plt.text(1, 1.13, rf'\textbf{{{"" if text_months == "ANNUAL" else f"{text_months} "}}}', fontsize=9, ha='right', va='top', transform=ax.transAxes)
    plt.text(1.04, 1.08, rf'\textbf{{{epoch_1[0]}–{epoch_1[1]}}}', fontsize=9, ha='right', va='top', transform=ax.transAxes)
    plt.text(.99, 1.03, r'\textbf{to}', fontsize=9, ha='right', va='top', transform=ax.transAxes)
    plt.text(1.04, .98, rf'\textbf{{{epoch_2[0]}–{epoch_2[1]}}}', fontsize=9, ha='right', va='top', transform=ax.transAxes)
    plt.text(.12, 1.07, rf'\textbf{{{f"{round(mean_diff_warming_value, 2):+.2f} hPa" if mean_diff_warming_value != 0 else "0 hPa"}}}',
             fontsize=9, rotation='horizontal', ha='right', va='top', transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='k', boxstyle='round,pad=0.2'))

    ### Save figure
    plt.subplots_adjust(top=.98, bottom=.03, left=.08, right=.92)
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_Global_SLP_Diff_{text_months}_{epoch_1[0]}–{epoch_1[1]}_{epoch_2[0]}–{epoch_2[1]}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')