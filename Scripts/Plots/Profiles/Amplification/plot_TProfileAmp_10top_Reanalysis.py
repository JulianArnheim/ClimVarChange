"""
Plot Temperature (T) amplification as in Arnheim et al. (2025) for reanalysis

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

def cosine_weighted_spatial_mean(data, coord=None):
    coord = coord or ('lat' if 'lat' in data.coords else 'latitude' if 'latitude' in data.coords else None)
    if not coord:
        raise ValueError("No 'lat' or 'latitude' coordinate found in the data array.")
    if 'lat' in data.coords and 'latitude' in data.coords:
        raise ValueError("Both 'lat' and 'latitude' coordinates exist.")
    weights = np.cos(np.deg2rad(data[coord]))
    return (data * weights).sum(dim=coord) / (data.notnull() * weights).sum(dim=coord)

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Profile/Amplification/Global/10 top/')

### Years and epochs
year_min, year_max = 1940, 2025
epoch_1 = (1941, 1970)
epoch_2 = (1991, 2020)

print(f'\n---- Arctic Amplification ----\n')

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

### Month and sector settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}
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
        text_months = sel_month.split()[1]
        text_longitudes = sel_longitude.split()[1]

        ### Figure
        print(f'{text_months} | {text_longitudes}')

        output_name = f'ERA5_Global_{epoch_1[0]}-{epoch_1[1]}_to_{epoch_2[0]}-{epoch_2[1]}_TProfileAmp_{text_months}_{text_longitudes}.nc'
        output_path = os.path.join(data_dir, output_name)
        if os.path.exists(output_path):
            print(f'File exists: {output_name}')
            mean_profile_amp = xr.open_dataset(output_path)

        else:
            print(f'File being made: {output_name}')

            ### Month and sector specification
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

            ### Amplification calculation from the zonal mean change in the latitude-pressure plane
            period_1 = datat_mean_profile.sel(year=slice(epoch_1[0], epoch_1[1])).chunk({'latitude': 15})
            period_2 = datat_mean_profile.sel(year=slice(epoch_2[0], epoch_2[1])).chunk({'latitude': 15})
            mean_profile_1 = period_1.mean(dim='year')
            mean_profile_2 = period_2.mean(dim='year')
            mean_diff_profile = mean_profile_2 - mean_profile_1

            ### All-latitude mean change by pressure level
            amp_ref = cosine_weighted_spatial_mean(mean_profile_2) - cosine_weighted_spatial_mean(mean_profile_1)

            ### Subtract reference change by height to isolate latitude dependence
            mean_profile_amp = mean_diff_profile - amp_ref

            ### Clear intermediates
            del datat_mean_profile, period_1, period_2, mean_diff_profile, amp_ref
            gc.collect()

            ### Save processed T amplification data
            encoding = {var: {'zlib': True, 'complevel': 3, 'dtype': 'float32'} for var in mean_profile_amp.data_vars}
            mean_profile_amp.to_netcdf(output_path, encoding=encoding)
            print(f'Saved {output_name}')

        ### Figure design
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
        plt.rc('savefig', facecolor='w')
        plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='k')
        plt.rc('xtick', color='dimgray', labelcolor='k')
        plt.rc('ytick', color='dimgray', labelcolor='k')

        ### Create figure
        fig, ax = plt.subplots(figsize=(9, 5))
        plt.xticks(fontname='Avant Garde')
        plt.yticks(fontname='Avant Garde')

        ### Spines
        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_color('dimgray')
            ax.spines[side].set_linewidth(2)

        ### Ticks and domain
        xticks = np.arange(-90, 91, 15)
        zscale = np.array([1000, 700, 500, 300, 200, 100, 50, 30, 10])
        ax.tick_params(axis='both', direction='out', which='major', pad=3, width=2, color='dimgray')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.yscale('log')
        plt.minorticks_off()
        ax.invert_yaxis()
        ax.set_xticks(xticks)
        ax.set_yticks(zscale)
        ax.set_xticklabels([f'{tick}°' for tick in xticks], fontname='Avant Garde', fontsize=11)
        ax.set_yticklabels([str(p) for p in zscale], fontname='Avant Garde', fontsize=11)
        ax.set(xlim=[-90, 90], ylim=[1000, 10])

        ### Custom color
        cmap = plt.get_cmap('RdBu_r')
        levels = np.arange(-2, 2.25, .25)
        middle_positions = np.linspace(.05, .95, len(levels))
        value_positions = np.concatenate([[levels[0] - .5], levels, [levels[-1] + .5]])
        color_positions = np.concatenate([[.01], middle_positions, [.99]])
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_RdBu', [cmap(p) for p in color_positions])
        norm = mcolors.BoundaryNorm(value_positions, custom_cmap.N)

        ### Plot
        im = ax.contourf(mean_profile_amp['latitude'], mean_profile_amp['level'], mean_profile_amp.t, norm=norm, cmap=custom_cmap, levels=levels, extend='both')

        ### Colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='horizontal', aspect=40, pad=.15, shrink=.8)
        cbar.ax.tick_params(labelsize=12, labelright=True, labelleft=False, right=True, left=False, color='dimgray')
        cbar.ax.set_xticks(np.linspace(-2, 2, 9))
        cbar.ax.set_xticklabels(['-2', '-1.5', '-1', '-.5', '0', '+.5', '+1', '+1.5', '+2'], color='k')
        cbar.ax.minorticks_off()
        cbar.ax.set_xlabel(r'\textbf{Difference in Temperature [°C]}', fontsize=16, color='k', labelpad=6)

        ### Title and labels
        plt.title(rf'\textbf{{{sel_longitude[3:]} TEMPERATURE $(T)$ AMPLIFICATION}}', color='k', fontsize=18, y=1.06)
        plt.xlabel(r'\textbf{Latitude [°N]}', color='k', fontsize=15)
        plt.ylabel(r'\textbf{Pressure [hPa]}', color='k', fontsize=15)

        ### Annotations
        season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
        season_text = rf'\textbf{{{season_prefix}{epoch_1[0]}–{epoch_1[1]} to {epoch_2[0]}–{epoch_2[1]}}}'
        plt.text(1, -.11, season_text, fontsize=11, rotation='horizontal', ha='right', va='top', color='k', transform=ax.transAxes)
        annotation_text = r'\textit{Relative to All-Latitude (90°S–90°N) Temperature Change by Pressure Level}'
        plt.text(0, -.12, annotation_text, fontsize=5, rotation='horizontal', ha='left', va='top', color='k', transform=ax.transAxes)

        ### Save figure
        plt.subplots_adjust(bottom=.05)
        plt.savefig(os.path.join(fig_dir, f'Reanalysis_TProfile_Amplevs_10top_{text_months}_{text_longitudes}.png'), dpi=1000)
        plt.close()

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')