"""
Plot Temperature (T) profile from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import matplotlib.colors as mcolors, matplotlib.pyplot as plt
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
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Profile/Historical/NH/10 top/T/')

### Years and epoch
year_min, year_max, epoch_1 = 1940, 2025, (1981, 2010)

print(f'\n---- Atmospheric Temperature Profile ----\n')

### Load data
datat = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_Global_{start_year}-{end_year}_TProfile_Monthly.nc')
    if os.path.exists(filename):
        ds = xr.open_dataset(filename, chunks={'time': 12, 'latitude': 16, 'level': 10}).astype(np.float32)
        datat.append(ds)
        print(f'Loaded: {start_year}–{end_year}')
if datat:
    datat = xr.concat(datat, dim='time').sortby('time')
else:
    raise FileNotFoundError('No ERA5_Global_*_TProfile_Monthly.nc files found.')
print()

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### Figure
    print(f'{text_months}')

    ### Calculate and save annual zonal means
    full_year_min = year_min + 1 if text_months == 'DJF' else year_min
    annual_zonal_mean_datat_filename = os.path.join(data_dir, f'ERA5_Global_{full_year_min}-{year_max}_TProfile_{text_months}_NH_AnnualZonalMeans.nc')
    if os.path.exists(annual_zonal_mean_datat_filename):
        annual_zonal_mean_datat = xr.open_dataset(annual_zonal_mean_datat_filename)
        print(f'Using Previously Computed ERA5 Annual Zonal Means')

    else:
        ### Month specification
        datat_season = datat.sel(time=datat['time'].dt.month.isin(mons))

        ### Process seasonal mean
        if text_months == 'DJF':
            winter_year = xr.where(datat_season['time'].dt.month == 12, datat_season['time'].dt.year + 1, datat_season['time'].dt.year)
            annual_zonal_mean_datat = datat_season.mean(dim='longitude').assign_coords(winter_year=('time', winter_year.data))
            annual_zonal_mean_datat = annual_zonal_mean_datat.groupby('winter_year').mean(dim='time')
            annual_zonal_mean_datat = annual_zonal_mean_datat.sel(winter_year=slice(full_year_min, year_max)).rename({'winter_year': 'year'})
        else:
            annual_zonal_mean_datat = datat_season.mean(dim='longitude').groupby('time.year').mean(dim='time')

        ### Save processed data
        annual_zonal_mean_datat.to_netcdf(annual_zonal_mean_datat_filename)
        print(f'Saved {annual_zonal_mean_datat_filename}, the ERA5 Annual Zonal Means\n')

    ### Mean profile
    mean_profile = annual_zonal_mean_datat.sel(year=slice(epoch_1[0], epoch_1[1])).mean(dim='year')

    ### Figure design
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='w')
    plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='k')
    plt.rc('xtick', color='dimgray', labelcolor='k')
    plt.rc('ytick', color='dimgray', labelcolor='k')

    ### Create figure
    fig, ax = plt.subplots(figsize=(9, 5))

    ### Spines
    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_color('dimgray')
        ax.spines[side].set_linewidth(2)
        ax.spines[side].set_zorder(10)

    ### Ticks and domain
    xticks, zscale = np.arange(0, 91, 5), np.array([1000, 700, 500, 300, 200, 100, 50, 30, 10])
    ax.tick_params(axis='both', direction='out', which='major', pad=3, width=2, color='dimgray')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.yscale('log')
    plt.minorticks_off()
    plt.gca().invert_yaxis()
    ax.invert_yaxis()
    ax.set_xticks(xticks)
    ax.set_yticks(zscale)
    ax.set_xticklabels([f'{tick}°' for tick in xticks], fontname='Avant Garde', fontsize=11)
    ax.set_yticklabels([str(p) for p in zscale], fontname='Avant Garde', fontsize=11)
    ax.set(xlim=[0, 90], ylim=[1000, 10])

    ### Colorbar settings
    norm = mcolors.TwoSlopeNorm(vmin=-90, vcenter=0, vmax=30)

    ### Plot
    im = ax.contourf(mean_profile['latitude'], mean_profile['level'], mean_profile.t, levels=np.arange(-90, 31, 5), cmap=plt.get_cmap('RdBu_r'), norm=norm, extend='both', zorder=0)

    ### Colorbar
    ticks = np.arange(-90, 31, 10)
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', aspect=40, pad=.15, shrink=.8)
    cbar.ax.tick_params(labelsize=12, labelright=True, labelleft=False, right=True, left=False, color='dimgray')
    cbar.set_ticks(ticks, labels=[str(i) for i in ticks])
    cbar.ax.minorticks_off()
    cbar.ax.set_xlabel(r'\textbf{Temperature [°C]}', fontsize=16, color='k', labelpad=6)

    ### Title, labels, and annotation
    plt.title(r'\textbf{TEMPERATURE $(T)$}', color='k', fontsize=20, y=1.04)
    plt.xlabel(r'\textbf{Latitude [°N]}', color='k', fontsize=15)
    plt.ylabel(r'\textbf{Pressure [hPa]}', color='k', fontsize=15)
    season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
    plt.text(1, 1.05, rf'\textbf{{{season_prefix}{epoch_1[0]}–{epoch_1[1]}}}', fontsize=10, rotation='horizontal', ha='right', va='top', color='k', transform=ax.transAxes)

    ### Save figure
    plt.subplots_adjust(bottom=.05)
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_{epoch_1[0]}–{epoch_1[1]}_TProfile_NH_10top_{text_months}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')