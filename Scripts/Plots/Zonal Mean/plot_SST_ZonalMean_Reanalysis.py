"""
Plot zonal mean Sea Surface Temperature (SST) Hovmöller

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import matplotlib as mpl, matplotlib.colors as mcolors, matplotlib.pyplot as plt
import numpy as np
import os
import time
import xarray as xr

mpl.use('Agg')  # non-interactive backend to avoid tkinter issues

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 0))
        else:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Zonal Mean/SST/')

### Years and epoch
year_min, year_max = 1940, 2025
epoch = (1981, 2010)

print(f'\n---- Zonal Mean Warming ----\n')

### Load data
filename = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_SST_Monthly.nc')
datat = xr.open_dataset(filename)['sst']

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
        text_months, text_longitudes = sel_month.split()[1], sel_longitude.split()[1]

        ### Figure
        print(f'{text_months} | {text_longitudes}')

        ### Select desired months and sector
        datat_sel = datat.sel(time=datat['time'].dt.month.isin(mons))
        datat_sel = datat_sel.where(lon_func(datat_sel.longitude), drop=True)

        ### Calculate annual means and anomalies
        if text_months == 'DJF':
            month, year = datat_sel.time.dt.month, datat_sel.time.dt.year
            winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
            zonal_mean_datat = datat_sel.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
            zonal_mean_datat = zonal_mean_datat.mean(dim='longitude').groupby('winter_year').mean(dim='time')
            zonal_mean_datat = zonal_mean_datat.sel(winter_year=slice(year_min + 1, year_max)).rename({'winter_year': 'year'})
        else:
            zonal_mean_datat = datat_sel.mean(dim='longitude').groupby('time.year').mean(dim='time')
        zonal_mean_datat_clim = zonal_mean_datat.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')
        zonal_mean_datat_anom = zonal_mean_datat - zonal_mean_datat_clim

        ### Prepare heatmap
        latitude, year = zonal_mean_datat_anom['latitude'], zonal_mean_datat_anom['year']
        SST_anom = zonal_mean_datat_anom.T

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
        fig, ax = plt.subplots(figsize=(12, 7))

        ### Adjust axes
        adjust_spines(ax, ['left', 'right', 'bottom', 'top'])
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

        ### Plot
        levels = np.arange(-2, 2.1, .25)
        value_positions = np.concatenate([[levels[0] - .5], levels, [levels[-1] + .5]])
        color_positions = np.concatenate([[.01], np.linspace(.05, .95, len(levels)), [.99]])
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_RdBu', [plt.get_cmap('RdBu_r')(p) for p in color_positions])
        cf = ax.contourf(year, latitude, SST_anom, levels=levels, cmap=custom_cmap, extend='both', norm=mcolors.BoundaryNorm(value_positions, custom_cmap.N))
        if np.any(np.isnan(SST_anom)):
            ax.contourf(year, latitude, np.isnan(SST_anom), levels=[.5, 1], colors=['lightgray'], extend='neither', zorder=1)

        ### Colorbar
        cbar_ticks = np.arange(-2, 2.1, 1)
        cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.14, shrink=.72, extend='both')
        cbar.set_label(r'\textbf{Sea Surface Temperature Anomaly [°C]}', fontsize=18, color='k', labelpad=12, fontname='Avant Garde')
        cbar.set_ticks(cbar_ticks)
        cbar.ax.tick_params(which='minor', length=0, colors='dimgray'); cbar.ax.tick_params(colors='dimgray')
        cbar.ax.set_xticklabels([f'{int(tick)}' if tick == 0 else f'{int(tick):+d}' for tick in cbar_ticks], fontsize=12, fontname='Avant Garde', color='k')

        ### Ticks
        xtick_labels, ytick_labels = np.arange(year_min, year_max + 1, 10), np.arange(-90, 91, 15)
        ax.set_xticks(xtick_labels); ax.set_yticks(ytick_labels)
        ax.set_xticklabels([str(xtick) for xtick in xtick_labels], fontsize=12, color='k')
        ax.set_yticklabels([f"{abs(lat)}°{'N' if lat > 0 else 'S' if lat < 0 else ''}" for lat in ytick_labels], fontsize=12, color='k')
        ax.set_ylim(-90, 90)

        ### Title, labels, and annotation
        plt.title(rf'\textbf{{{text_months} {sel_longitude[3:]} MEAN WARMING}}', color='k', fontsize=20, y=1.05)
        plt.xlabel(r'\textbf{Year}', fontsize=16, color='k', labelpad=12, fontweight='bold', fontname='Avant Garde')
        plt.ylabel(r'\textbf{Latitude}', fontsize=16, color='k', labelpad=12)
        plt.text(year_max, 93, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}}}', fontsize=11, rotation='horizontal', ha='right', color='k')

        ### Save figure
        plt.subplots_adjust(top=.88, bottom=.02, left=.1, right=.95)
        fig_path = os.path.join(fig_dir, f'Reanalysis_SST_ZonalMean_{text_months}_{text_longitudes}_{year_min}–{year_max}.png')
        plt.savefig(fig_path, dpi=1000)
        plt.close()

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')