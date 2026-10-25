"""
Plot heatmap of Blocking Index (BLO) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import cartopy.crs as ccrs, cartopy.feature as cfeature, cartopy.util as cutil
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
fig_dir = os.path.join(base_dir, 'Figures/Spatial/NH/BLO/')

### Years
year_min, year_max = 1940, 2025

### Epochs
epochs = [(year_min, year_max), (1981, 2010)]

print(f'\n---- Blocking Index (BLO) Heatmap ----\n')

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through epochs
for start_year, end_year in epochs:

    ### Loop through seasons
    for sel_month, mons in months_options.items():

        ### Season name
        text_months = sel_month.split()[1]

        ### First valid year
        first_year = start_year + 1 if (text_months == 'DJF' and start_year == year_min) else start_year

        ### Load BLO data
        BLO_datasets = []
        for dec_start in range((start_year // 10) * 10, (end_year // 10) * 10 + 1, 10):
            dec_end = min(dec_start + 9, year_max)
            filename = os.path.join(data_dir, f'ERA5_BLO_1d_{dec_start}-{dec_end}_Daily.nc')
            if os.path.exists(filename):
                BLO_datasets.append(xr.open_dataset(filename))
                print(f'Loaded: {dec_start}–{dec_end}')
            else:
                print(f'File not found: {dec_start}–{dec_end}')
        BLOs = xr.concat(BLO_datasets, dim='time')
        print(f'Concatenated BLOs')

        ### Apply 5-day persistence criterion
        window_end = (BLOs.sortby('time').rolling(time=5, min_periods=5).sum() >= 5).astype(np.int8)  # 1 on the last day of each fully blocked 5-day window
        BLOs_5d = (window_end.rolling(time=5, min_periods=1).max().shift(time=-4, fill_value=0) >= 1).astype(np.int8)  # extend the flag back over the window

        ### Process data, labelling DJF days by the January year
        BLO = BLOs_5d.sel(time=BLOs_5d.time.dt.month.isin(mons))
        if text_months == 'DJF':
            winter_year = xr.where(BLO.time.dt.month == 12, BLO.time.dt.year + 1, xr.where(BLO.time.dt.month.isin([1, 2]), BLO.time.dt.year, np.nan))
            BLO = BLO.assign_coords(year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
        else:
            BLO = BLO.assign_coords(year=('time', BLO.time.dt.year.data))
        BLO = BLO.sel(time=(BLO.year >= first_year) & (BLO.year <= end_year))

        ### Fraction of days with at least one blocked grid point, then annual means
        BLO_fraction = BLO['BLO'].max(dim=('latitude', 'longitude')).mean(dim='time').item() * 100
        BLO = BLO.groupby('year').mean(dim='time')

        ### Blocking frequency
        BLO_field = BLO.mean(dim='year')['BLO']
        print(f'\n{text_months} {first_year}–{end_year}:')
        BLO_min = BLO_field.min().values
        BLO_mean = BLO_field.mean().values
        BLO_max = BLO_field.max().values
        stats = {'Minimum': BLO_min * 100, 'Mean': BLO_mean * 100, 'Maximum': BLO_max * 100}
        for name, val in stats.items():
            print(f'{name:>8}: {val:5.2f}%')

        ### Figure design
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
        plt.rc('savefig', facecolor='w')
        plt.rc('xtick', color='k')
        plt.rc('ytick', color='k')
        plt.rc('axes', facecolor='w', labelcolor='k')

        ### Create figure
        fig, ax = plt.subplots(figsize=(6.5, 3.25), subplot_kw={'projection': ccrs.Robinson()})
        ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
        ax.set_xticks([])
        ax.coastlines()
        ax.add_feature(cfeature.LAND, facecolor='none', linewidth=.5, zorder=2)
        ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
        for x in [(-130, 'AM'), (-40, 'AT'), (55, 'AS'), (135, 'PA'), (-175, 'PA')]:
            ax.text(x[0], 2, x[1], transform=ccrs.PlateCarree(), fontsize=12, color='b')

        ### Gridlines
        ax.gridlines(draw_labels=True, xlocs=[-140, -50, 40, 130], ylocs=[35, 75], linestyle='--', color='k')
        ax.gridlines(xlocs=[-140, -50, 40, 130], ylocs=[35, 75], linestyle='--', color='k')

        ### Interpolate onto clean target longitude grid: [-180 to 180) by 2.5°
        lon_target, BLO_field = np.arange(-180, 180, 2.5), BLO_field.assign_coords(longitude=((BLO_field.longitude + 180) % 360) - 180).sortby('longitude')
        BLO_cyclic, lon_cyclic = cutil.add_cyclic_point(BLO_field.interp(longitude=lon_target), coord=lon_target)  # cyclic point for smooth wraparound
        BLO_cyclic = np.maximum(BLO_cyclic, 0)  # ensure no negative values

        ### Colors
        cmap, levels = plt.get_cmap('YlGnBu', lut=192), np.linspace(0, 10, 11)

        ### Plot
        cf = ax.contourf(lon_cyclic, BLO_field.latitude, BLO_cyclic * 100, levels=levels, cmap=cmap, vmin=0, vmax=10, extend='max', transform=ccrs.PlateCarree())

        ### Colorbar
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', aspect=25, pad=.18, shrink=.5, extend='max')
        cbar.set_label(r'Blocking Frequency [\%]', fontsize=12, labelpad=8)
        cbar.set_ticks(np.arange(0, 11, 2))
        cbar.set_ticklabels([f'{i}' for i in range(0, 11, 2)])
        cbar.ax.tick_params(labelsize=8)

        ### Title
        plt.title(r'\textbf{Blocking Index (BLO)}', color='k', fontsize=20, y=1.22)

        ### Annotations
        season_label = '' if text_months == 'ANNUAL' else f'{text_months} '
        ax.text(.83, 1.13, rf'\textbf{{{season_label}{first_year}–{end_year}}}', transform=ax.transAxes, fontsize=10, verticalalignment='bottom')
        ax.text(-.02, 1.13, rf'{BLO_fraction:.0f}\% of days' if BLO_fraction == 100 else rf'{BLO_fraction:.1f}\% of days', transform=ax.transAxes, fontsize=10, ha='left', va='bottom')

        ### Save figure
        plt.subplots_adjust(top=.78, bottom=.12, left=.08, right=.92)
        plt.savefig(os.path.join(fig_dir, f'Reanalysis_BLO_Heatmap_{text_months}_{first_year}–{end_year}.png'), dpi=1000)
        plt.close()

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')