"""
Plot change in Temperature (T) and Zonal Wind (U) profile for reanalysis

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

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Profile/Diffs/NH/100 top/TU/')

### Years and epochs
year_min, year_max = 1940, 2025
epoch_1, epoch_2 = (1941, 1970), (1991, 2020)

print(f'\n---- Global Warming & Zonal Wind Response ----\n')

### Load T data
datat = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_Global_{start_year}-{end_year}_TProfile_Monthly.nc')
    if os.path.exists(filename):
        ds = xr.open_dataset(filename, chunks={}).astype(np.float32)
        datat.append(ds)
        print(f'Loaded T: {start_year}–{end_year}')
if datat:
    datat = xr.concat(datat, dim='time').sortby('time')
else:
    raise FileNotFoundError('No ERA5_Global_*_TProfile_Monthly.nc files found.')
print()

### Load U data
datau = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_Global_{start_year}-{end_year}_UProfile_Monthly.nc')
    if os.path.exists(filename):
        ds = xr.open_dataset(filename, chunks={}).astype(np.float32)
        datau.append(ds)
        print(f'Loaded U: {start_year}–{end_year}')
if datau:
    datau = xr.concat(datau, dim='time').sortby('time')
else:
    raise FileNotFoundError('No ERA5_Global_*_UProfile_Monthly.nc files found.')
print()

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Sector settings
longitudes_options = {'1. NH (180°W–180°E)': lambda lon: (lon >= 0) & (lon < 360),
                      '2. AM (140°W–50°W)': lambda lon: (lon >= 360 - 140) & (lon < 360 - 50),  # -140 to -50
                      '3. AT (50°W–40°E)': lambda lon: ((lon >= 360 - 50) & (lon < 360)) | ((lon >= 0) & (lon <= 40)),  # -50 to 40
                      '4. AS (40°E–130°E)': lambda lon: (lon >= 40) & (lon <= 130),  # 40 to 130
                      '5. PA (130°E–140°W)': lambda lon: (lon >= 130) & (lon < 360 - 140)}  # 130 to -140

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### Loop through each sector
    for sel_longitude, lon_func in longitudes_options.items():

        ### String corrections
        text_months, text_longitudes = sel_month.split()[1], sel_longitude.split()[1]

        ### Figure
        print(f'{text_months} | {text_longitudes}')

        ### Load T data
        T_output_name = f'ERA5_Global_{epoch_1[0]}-{epoch_1[1]}_to_{epoch_2[0]}-{epoch_2[1]}_TProfile_{text_months}_{text_longitudes}.nc'
        T_output_path = os.path.join(data_dir, T_output_name)

        ### Load U data
        U_output_name = f'ERA5_Global_{epoch_1[0]}-{epoch_1[1]}_to_{epoch_2[0]}-{epoch_2[1]}_UProfile_{text_months}_{text_longitudes}.nc'
        U_output_path = os.path.join(data_dir, U_output_name)

        ### Initializing presence of processed data
        diff_t_means, diff_u_means = None, None

        ### Checking for processed T data before computing
        if os.path.exists(T_output_path):
            print(f'File exists: {T_output_name}')
            diff_t_means = xr.open_dataset(T_output_path)

        ### Checking for processed U data before computing
        if os.path.exists(U_output_path):
            print(f'File exists: {U_output_name}')
            diff_u_means = xr.open_dataset(U_output_path)

        ### Computing if either processed file does not exist
        if diff_t_means is None or diff_u_means is None:

            if diff_t_means is None:

                ### Process T data
                datat_sel = datat.sel(time=datat['time'].dt.month.isin(mons))
                datat_sel = datat_sel.where(lon_func(datat_sel.longitude), drop=True)

                ### Process seasonal mean
                if text_months == 'DJF':
                    winter_year = xr.where(datat_sel['time'].dt.month == 12, datat_sel['time'].dt.year + 1, datat_sel['time'].dt.year)
                    datat_mean_profile = datat_sel.mean(dim='longitude').assign_coords(winter_year=('time', winter_year.data))
                    datat_mean_profile = datat_mean_profile.groupby('winter_year').mean(dim='time')
                    datat_mean_profile = datat_mean_profile.sel(winter_year=slice(year_min + 1, year_max)).rename({'winter_year': 'year'})
                else:
                    datat_mean_profile = datat_sel.mean(dim='longitude').groupby('time.year').mean(dim='time')

                ### Compute difference in means
                datat_profile_1 = datat_mean_profile.sel(year=slice(epoch_1[0], epoch_1[1])).chunk({'latitude': 15})
                datat_profile_2 = datat_mean_profile.sel(year=slice(epoch_2[0], epoch_2[1])).chunk({'latitude': 15})
                diff_t_means = datat_profile_2.mean(dim='year') - datat_profile_1.mean(dim='year')

                ### Clear intermediates
                del datat_mean_profile, datat_profile_1, datat_profile_2
                gc.collect()

                ### Save processed T data
                encoding = {var: {'zlib': True, 'complevel': 3, 'dtype': 'float32'} for var in diff_t_means.data_vars}
                diff_t_means.to_netcdf(T_output_path, encoding=encoding)
                print(f'Saved {T_output_name}')

            if diff_u_means is None:

                ### Process U data
                datau_sel = datau.sel(time=datau['time'].dt.month.isin(mons))
                datau_sel = datau_sel.where(lon_func(datau_sel.longitude), drop=True)

                ### Process seasonal mean
                if text_months == 'DJF':
                    winter_year = xr.where(datau_sel['time'].dt.month == 12, datau_sel['time'].dt.year + 1, datau_sel['time'].dt.year)
                    datau_mean_profile = datau_sel.mean(dim='longitude').assign_coords(winter_year=('time', winter_year.data))
                    datau_mean_profile = datau_mean_profile.groupby('winter_year').mean(dim='time')
                    datau_mean_profile = datau_mean_profile.sel(winter_year=slice(year_min + 1, year_max)).rename({'winter_year': 'year'})
                else:
                    datau_mean_profile = datau_sel.mean(dim='longitude').groupby('time.year').mean(dim='time')

                ### Compute difference in means
                datau_profile_1 = datau_mean_profile.sel(year=slice(epoch_1[0], epoch_1[1])).chunk({'latitude': 15})
                datau_profile_2 = datau_mean_profile.sel(year=slice(epoch_2[0], epoch_2[1])).chunk({'latitude': 15})
                diff_u_means = datau_profile_2.mean(dim='year') - datau_profile_1.mean(dim='year')

                ### Clear intermediates
                del datau_mean_profile, datau_profile_1, datau_profile_2
                gc.collect()

                ### Save processed U data
                encoding = {var: {'zlib': True, 'complevel': 3, 'dtype': 'float32'} for var in diff_u_means.data_vars}
                diff_u_means.to_netcdf(U_output_path, encoding=encoding)
                print(f'Saved {U_output_name}')

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

        ### Ticks
        xticks = np.arange(0, 91, 5)
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
        ax.set_xticklabels([f'{tick}°' for tick in xticks], fontname='Avant Garde', fontsize=11)
        ax.set_yticklabels([str(p) for p in zscale], fontname='Avant Garde', fontsize=11)

        ### Domain
        ax.set(xlim=[0, 90], ylim=[1000, 100])

        ### Custom color (T)
        cmap = plt.get_cmap('RdBu_r')
        levels = np.arange(-2, 2.25, .25)
        value_positions = np.concatenate([[levels[0] - .5], levels, [levels[-1] + .5]])
        norm = mcolors.BoundaryNorm(value_positions, cmap.N)

        ### Plot (T)
        im = ax.contourf(diff_t_means['latitude'], diff_t_means['level'], diff_t_means.t, norm=norm, cmap=cmap, levels=levels,
                         extend='both')

        ### Colorbar (T)
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', aspect=40, pad=.15, shrink=.8)
        cbar.ax.tick_params(labelsize=12, labelright=True, labelleft=False, right=True, left=False, color='dimgray')
        cbar.ax.set_xticks(np.linspace(-2, 2, 9))
        cbar.ax.set_xticklabels(['-2', '-1.5', '-1', '-.5', '0', '+.5', '+1', '+1.5', '+2'], color='k')
        cbar.ax.minorticks_off()
        cbar.ax.set_xlabel(r'\textbf{Difference in Temperature [°C] and Zonal Wind [m s\textsuperscript{-1}]}', fontsize=14, color='k', labelpad=6)

        ### Contours (U)
        contour_levels = np.concatenate([np.arange(-10, 0, 1), np.arange(1, 10, 1)])
        fmt = lambda x: (f'{x:+g}'.lstrip('0')) if x >= 0 else f'{x:g}'
        nonzero_levels = [l for l in contour_levels if l != 0]
        CS = ax.contour(diff_u_means['latitude'], diff_u_means['level'], diff_u_means['u'], levels=nonzero_levels, colors='k', linewidths=1)
        clabels = ax.clabel(CS, CS.levels, inline=True, fontsize=7, fmt=fmt, colors='k', manual=False, rightside_up=True, use_clabeltext=True)

        ### Title
        plt.title(rf'\textbf{{{sel_longitude[3:]} TEMPERATURE AND ZONAL WIND DIFFERENCE $(\Delta T, \Delta U)$}}', color='k', fontsize=16, y=1.06)

        ### Labels
        plt.xlabel(r'\textbf{Latitude [°N]}', color='k', fontsize=15); plt.ylabel(r'\textbf{Pressure [hPa]}', color='k', fontsize=15)

        ### Annotation
        season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
        season_text = rf'\textbf{{{season_prefix}{epoch_1[0]}–{epoch_1[1]} to {epoch_2[0]}–{epoch_2[1]}}}'
        plt.text(1, -.12, season_text, fontsize=10, rotation='horizontal', ha='right', va='top', transform=ax.transAxes)

        ### Save figure
        plt.subplots_adjust(bottom=.05)
        plt.savefig(os.path.join(fig_dir, f'Reanalysis_TUProfileDiff_NH_100top_{text_months}_{text_longitudes}.png'), dpi=1000)
        plt.close()

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')