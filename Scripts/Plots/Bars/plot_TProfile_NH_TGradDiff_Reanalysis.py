"""
Plot the change between epochs in the meridional temperature difference (30–50°N minus 50–70°N mean) by level from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np
import os
import time
import xarray as xr

mpl.use('Agg')  # non-interactive backend to avoid tkinter issues

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

def mean_diff(data):
    subtrop_mean = cosine_weighted_spatial_mean(data.sel(latitude=lat_ranges['subtrop']))
    subpolar_mean = cosine_weighted_spatial_mean(data.sel(latitude=lat_ranges['subpolar']))
    return subtrop_mean - subpolar_mean

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Bars/')

### Years and epochs
year_min, year_max = 1940, 2025
epochs_1 = [(1941, 1970), (1951, 1980)]  # first time period options
epochs_2 = [(1981, 2010), (1991, 2020)]  # second time period options

print(f'\n---- Changing Meridional Temperature Gradient ----\n')

### Levels and latitudes
levs = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150]
lat_ranges = {'subtrop': slice(50, 30), 'subpolar': slice(70, 50)}

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='k')
plt.rc('xtick', color='dimgray', labelcolor='k')
plt.rc('ytick', color='dimgray', labelcolor='k')

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
print(f'Concatenated Temperature Profile\n')

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop to produce versions of the figure with different epochs
for epoch_1_start, epoch_1_end in epochs_1:
    for epoch_2_start, epoch_2_end in epochs_2:

        ### Loop through each set of months
        for sel_month, mons in months_options.items():

            ### String corrections
            text_months = sel_month.split()[1]

            print(f'{text_months} | ({epoch_1_start}–{epoch_1_end}) to ({epoch_2_start}–{epoch_2_end})')

            ### Process
            if text_months == 'DJF':
                datat_djf = datat.sel(time=datat['time'].dt.month.isin(mons))
                winter_year = xr.where(datat_djf['time'].dt.month == 12, datat_djf['time'].dt.year + 1, datat_djf['time'].dt.year)
                datat_level = datat_djf.assign_coords(winter_year=('time', winter_year.data)).groupby('winter_year').mean(dim='time').mean(dim='longitude').sel(level=levs)
                datat_level = datat_level.sel(winter_year=slice(year_min + 1, year_max)).rename({'winter_year': 'year'})
            else:
                datat_level = datat.sel(time=datat['time'].dt.month.isin(mons)).groupby('time.year').mean(dim='time').mean(dim='longitude').sel(level=levs)

            ### Compute difference in epochs
            epoch_1_datat, epoch_2_datat = datat_level.sel(year=slice(epoch_1_start, epoch_1_end)).mean(dim='year'), datat_level.sel(year=slice(epoch_2_start, epoch_2_end)).mean(dim='year')
            diff = (mean_diff(epoch_2_datat) - mean_diff(epoch_1_datat)).to_array().squeeze()

            ### Create figure
            fig, ax = plt.subplots(figsize=(3.5, 3))
            ax.invert_yaxis()

            ### Grid lines
            ax.axvline(x=0, color='dimgray', linestyle='-', linewidth=.5, zorder=4)
            for x in [-.5, .5]:
                ax.axvline(x=x, color='dimgray', linestyle='-', linewidth=.5, zorder=2)
            for pos in np.arange(-1, 1.1, .1):
                ax.axvline(x=pos, color='dimgray', linestyle='--', linewidth=.3, zorder=2)

            ### Plot
            ax.barh(diff['level'], diff.values, height=40, color=np.where(diff >= 0, 'r', 'b'), zorder=3)
            ax.set_position([ax.get_position().x0 + .02, ax.get_position().y0 + .01, ax.get_position().width, ax.get_position().height])

            ### Spines
            ax.spines['bottom'].set_color('dimgray')
            ax.spines['left'].set_color('dimgray')

            ### Ticks and domain
            ax.tick_params(axis='both', direction='out', pad=2, width=.8, length=2.5, color='dimgray')
            ax.set_xticks(np.arange(-1, 1.1, .5))
            ax.set_xticklabels([f'0' if tick == 0 else f'{int(tick)}' if abs(tick) == 1 else f'{tick:+.1f}'.replace('0.', '.') for tick in np.arange(-1, 1.1, .5)], fontsize=6, zorder=2)
            ax.set_yticks(diff['level'])
            ax.set_yticklabels(np.round(ax.get_yticks()).astype(int), fontsize=6)
            ax.set_xlim([-1, 1])

            ### Title and labels
            plt.suptitle(f'\\textbf{{Changing Meridional Temperature Gradient}}', fontsize=10, color='k', x=.54, y=.985)
            ax.set_xlabel(r'\textbf{$\Delta \bar{T}$ [°C]}', fontsize=9, labelpad=4)
            ax.set_ylabel(r'\textbf{Pressure [hPa]}', fontsize=9, labelpad=3)

            ### Annotation
            plt.text(.538, .928, f'{text_months} {epoch_1_start}–{epoch_1_end} to {text_months} {epoch_2_start}–{epoch_2_end}', fontsize=7, ha='center', va='top', transform=plt.gcf().transFigure)

            ### Save figure
            plt.savefig(os.path.join(fig_dir, f'Reanalysis_NH_TGradDiff_{text_months}_{epoch_1_start}–{epoch_1_end}_{epoch_2_start}–{epoch_2_end}.png'), dpi=1000)
            plt.close(fig)

            ### Progress
            print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')