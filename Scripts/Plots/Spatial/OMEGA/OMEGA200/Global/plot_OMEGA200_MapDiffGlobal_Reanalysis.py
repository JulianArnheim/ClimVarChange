"""
Plot the difference in the 200 hPa vertical pressure velocity (OMEGA200) between two epochs from ERA5 reanalysis

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
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Global/OMEGA/OMEGA200/Diff/')

### Years and epochs
year_min, year_max = 1940, 2025
epoch_1, epoch_2 = (1941, 1970), (1991, 2020)

print(f'\n---- Difference in 200 hPa Vertical Pressure Velocity (OMEGA200) ----\n')

### Load data
filename = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_OMEGA200_Monthly.nc')
datap = xr.open_dataset(filename)['OMEGA200']

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### Figure
    print(f'{text_months}')

    ### Process data
    OMEGA200 = datap.sel(time=datap['time'].dt.month.isin(mons))
    if text_months == 'DJF':
        month, year = OMEGA200.time.dt.month, OMEGA200.time.dt.year
        winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
        OMEGA200 = OMEGA200.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
        OMEGA200 = OMEGA200.groupby('winter_year').mean(dim='time').rename({'winter_year': 'year'})
    else:
        OMEGA200 = OMEGA200.groupby('time.year').mean(dim='time')

    ### Compute difference in means
    diff_means = OMEGA200.sel(year=slice(epoch_2[0], epoch_2[1])).mean(dim='year') - OMEGA200.sel(year=slice(epoch_1[0], epoch_1[1])).mean(dim='year')

    ### Spatial mean vertical pressure velocity change
    weighted_diff = diff_means.weighted(np.cos(np.deg2rad(diff_means.latitude)))
    mean_diff_omega = weighted_diff.mean(dim=('latitude', 'longitude'))
    mean_diff_omega_value = mean_diff_omega.compute().values.item()
    min_diff_omega = diff_means.min().values
    max_diff_omega = diff_means.max().values
    print(f'Global [Min/Mean/Max]: {min_diff_omega:+.4e} / {mean_diff_omega_value:+.4e} / {max_diff_omega:+.4e} Pa s⁻¹')

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
    cf = ax.contourf(x, y, diff_means, levels=np.linspace(-1e-2, 1e-2, 21), cmap=plt.get_cmap('BrBG_r', lut=192), vmin=-1e-2, vmax=1e-2, transform=ccrs.PlateCarree(), extend='both')

    ### Colorbar
    cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.1, shrink=.8, extend='both')
    cbar.ax.tick_params(labelsize=10, color='dimgray', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
    cbar.ax.set_xticks(np.linspace(-1e-2, 1e-2, 5)); cbar.ax.set_xticklabels(['-10', '-5', '0', '+5', '+10'])
    cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.set_xlabel(r'\textbf{Difference in 200 hPa Vertical Pressure Velocity [10\textsuperscript{-3} Pa s\textsuperscript{-1}]}', fontsize=11, color='k', labelpad=10)

    ### Title
    plt.title(r'\textbf{VERTICAL PRESSURE VELOCITY CHANGE}', color='k', fontsize=18, y=1.12)

    ### Annotations
    plt.text(1, 1.13, rf'\textbf{{{"" if text_months == "ANNUAL" else f"{text_months} "}}}', fontsize=9, ha='right', va='top', transform=ax.transAxes)
    plt.text(1.04, 1.08, rf'\textbf{{{epoch_1[0]}–{epoch_1[1]}}}', fontsize=9, ha='right', va='top', transform=ax.transAxes)
    plt.text(.99, 1.03, r'\textbf{to}', fontsize=9, ha='right', va='top', transform=ax.transAxes)
    plt.text(1.04, .98, rf'\textbf{{{epoch_2[0]}–{epoch_2[1]}}}', fontsize=9, ha='right', va='top', transform=ax.transAxes)
    plt.text(.15, 1.07, rf'\textbf{{{(m := f"{mean_diff_omega_value:.2e}".split("e"))[0]} × 10\textsuperscript{{{int(m[1])}}} Pa s\textsuperscript{{-1}}}}', fontsize=9, ha='right', va='top',
             transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='k', boxstyle='round,pad=0.35'))

    ### Save figure
    plt.subplots_adjust(top=.98, bottom=.03, left=.08, right=.92)
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_Global_OMEGA200_Diff_{text_months}_{epoch_1[0]}–{epoch_1[1]}_{epoch_2[0]}–{epoch_2[1]}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')