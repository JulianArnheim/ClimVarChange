"""
Plot Temperature (T) and Meridional Wind (V) profile from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import gc
import matplotlib.colors as mcolors, matplotlib.pyplot as plt
import numpy as np
import os
import time
import xarray as xr

def fmt(x):
    text = f'{x:+.1f}'.replace('+0.', '+.').replace('-0.', '-.')
    return text.rstrip('0').rstrip('.')

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Profile/Historical/Global/100 top/TV/')

### Years
year_min, year_max = 1940, 2025

### Epoch
epoch_1 = (1981, 2010)

print(f'\n---- Atmospheric Temperature & Meridional Wind Profile ----\n')

### Load T data
datat = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    T_filename = os.path.join(data_dir, f'ERA5_Global_{start_year}-{end_year}_TProfile_Monthly.nc')
    if os.path.exists(T_filename):
        ds = xr.open_dataset(T_filename, chunks={}).astype(np.float32)
        datat.append(ds)
        print(f'Loaded T: {start_year}–{end_year}')
if datat:
    datat = xr.concat(datat, dim='time').sortby('time')
else:
    raise FileNotFoundError('No ERA5_Global_*_TProfile_Monthly.nc files found.')
print()

### Load V data
datav = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    V_filename = os.path.join(data_dir, f'ERA5_Global_{start_year}-{end_year}_VProfile_Monthly.nc')
    if os.path.exists(V_filename):
        ds = xr.open_dataset(V_filename, chunks={}).astype(np.float32)
        datav.append(ds)
        print(f'Loaded V: {start_year}–{end_year}')
if datav:
    datav = xr.concat(datav, dim='time').sortby('time')
else:
    raise FileNotFoundError('No ERA5_Global_*_VProfile_Monthly.nc files found.')
print()

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### Figure
    print(f'{text_months}')

    ### ----- TEMPERATURE (T) -----
    full_year_min = year_min + 1 if text_months == 'DJF' else year_min
    annual_zonal_mean_datat_filename = os.path.join(data_dir, f'ERA5_Global_{full_year_min}-{year_max}_TProfile_{text_months}_NH_AnnualZonalMeans.nc')
    if os.path.exists(annual_zonal_mean_datat_filename):
        annual_zonal_mean_datat = xr.open_dataset(annual_zonal_mean_datat_filename)
        print(f'Using Previously Computed ERA5 T Annual Zonal Means')

    else:
        ### Process T data
        datat_season = datat.sel(time=datat['time'].dt.month.isin(mons))

        ### Process seasonal mean
        if text_months == 'DJF':
            winter_year = xr.where(datat_season['time'].dt.month == 12, datat_season['time'].dt.year + 1, datat_season['time'].dt.year)
            annual_zonal_mean_datat = datat_season.mean(dim='longitude').assign_coords(winter_year=('time', winter_year.data))
            annual_zonal_mean_datat = annual_zonal_mean_datat.groupby('winter_year').mean(dim='time')
            annual_zonal_mean_datat = annual_zonal_mean_datat.sel(winter_year=slice(year_min + 1, year_max)).rename({'winter_year': 'year'})
        else:
            annual_zonal_mean_datat = datat_season.mean(dim='longitude').groupby('time.year').mean(dim='time')

        ### Save processed data
        annual_zonal_mean_datat.to_netcdf(annual_zonal_mean_datat_filename)
        print(f'Saved {annual_zonal_mean_datat_filename}, the ERA5 T Annual Zonal Means')

    ### ----- MERIDIONAL WIND (V) -----
    annual_zonal_mean_datav_filename = os.path.join(data_dir, f'ERA5_Global_{full_year_min}-{year_max}_VProfile_{text_months}_NH_AnnualZonalMeans.nc')
    if os.path.exists(annual_zonal_mean_datav_filename):
        annual_zonal_mean_datav = xr.open_dataset(annual_zonal_mean_datav_filename)
        print(f'Using Previously Computed ERA5 V Annual Zonal Means')

    else:
        ### Process V data
        datav_season = datav.sel(time=datav['time'].dt.month.isin(mons))

        ### Process seasonal mean
        if text_months == 'DJF':
            winter_year = xr.where(datav_season['time'].dt.month == 12, datav_season['time'].dt.year + 1, datav_season['time'].dt.year)
            annual_zonal_mean_datav = datav_season.mean(dim='longitude').assign_coords(winter_year=('time', winter_year.data))
            annual_zonal_mean_datav = annual_zonal_mean_datav.groupby('winter_year').mean(dim='time')
            annual_zonal_mean_datav = annual_zonal_mean_datav.sel(winter_year=slice(year_min + 1, year_max)).rename({'winter_year': 'year'})
        else:
            annual_zonal_mean_datav = datav_season.mean(dim='longitude').groupby('time.year').mean(dim='time')

        ### Save processed data
        annual_zonal_mean_datav.to_netcdf(annual_zonal_mean_datav_filename)
        print(f'Saved {annual_zonal_mean_datav_filename}, the ERA5 V Annual Zonal Means')

    ### Process T data
    T_selected = annual_zonal_mean_datat.sel(year=slice(epoch_1[0], epoch_1[1]))
    T_mean_profile = T_selected.mean(dim='year')

    ### Process V data
    V_selected = annual_zonal_mean_datav.sel(year=slice(epoch_1[0], epoch_1[1]))
    V_mean_profile = V_selected.mean(dim='year')

    ### Clear intermediates
    del annual_zonal_mean_datat, annual_zonal_mean_datav
    gc.collect()

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
    xticks = np.arange(-90, 91, 15)
    zscale = np.array([1000, 700, 500, 300, 200, 100])
    ax.tick_params(axis='both', direction='out', which='major', pad=3, width=2, color='dimgray')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.yscale('log')
    plt.minorticks_off()
    plt.gca().invert_yaxis()
    ax.invert_yaxis()
    ax.set_xticks(xticks)
    ax.set_yticks(zscale)

    ### Tick labels
    ax.set_xticklabels([f'{tick}°' for tick in xticks], fontname='Avant Garde', fontsize=11)
    ax.set_yticklabels([str(p) for p in zscale], fontname='Avant Garde', fontsize=11)

    ### Domain
    ax.set(xlim=[-90, 90], ylim=[1000, 100])

    ### Color (T)
    cmap = plt.get_cmap('RdBu_r')
    levels = np.arange(-90, 31, 5)
    norm = mcolors.TwoSlopeNorm(vmin=-90, vcenter=0, vmax=30)

    ### Plot (T)
    im = ax.contourf(T_mean_profile['latitude'], T_mean_profile['level'], T_mean_profile.t, levels=levels, cmap=cmap, norm=norm, extend='both')

    ### Colorbar (T)
    levels = np.arange(-90, 31, 10)
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', aspect=40, pad=.15, shrink=.8)
    cbar.ax.tick_params(labelsize=12, labelright=True, labelleft=False, right=True, left=False, color='dimgray')
    cbar.set_ticks(levels)
    cbar.set_ticklabels([str(i) for i in levels])
    cbar.ax.minorticks_off()
    cbar.ax.set_xlabel(r'\textbf{Temperature [°C] and Meridional Wind [m s\textsuperscript{-1}]}', fontsize=16, color='k', labelpad=6)

    ### Contours (V)
    del_V = 1
    contour_levels = np.concatenate([np.arange(-100, 0, del_V), np.arange(del_V, 100, del_V)])
    contour_levels_nonzero = [l for l in contour_levels if l != 0]
    fmt = lambda x: f'{x:+.1f}'.replace('+0.', '+.').replace('-0.', '-.').rstrip('0').rstrip('.')
    CS = ax.contour(V_mean_profile['latitude'], V_mean_profile['level'], V_mean_profile['v'], levels=contour_levels_nonzero, colors='k', linewidths=1.25, zorder=5)
    clabels = ax.clabel(CS, CS.levels, inline=True, fontsize=8, fmt=fmt, colors='k', manual=False, rightside_up=True, use_clabeltext=True)

    ### Title
    plt.title(r'\textbf{TEMPERATURE $(T)$ AND MERIDIONAL WIND $(V)$}', color='k', fontsize=16, y=1.04)

    ### Labels
    plt.xlabel(r'\textbf{Latitude [°N]}', color='k', fontsize=15)
    plt.ylabel(r'\textbf{Pressure [hPa]}', color='k', fontsize=15)

    ### Annotations
    season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
    plt.text(1, 1.05, rf'\textbf{{{season_prefix}{epoch_1[0]}–{epoch_1[1]}}}', fontsize=10, rotation='horizontal', ha='right', va='top', color='k', transform=ax.transAxes)

    ### Save figure
    plt.subplots_adjust(bottom=.05)
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_{epoch_1[0]}–{epoch_1[1]}_TVProfile_100top_{text_months}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')