"""
Plot single day 500 hPa field and Blocking Index (BLO) from ERA5 reanalysis

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

def plot_map(date):

    print(f'Plot {date}')

    ### Load Z500 data (single file only)
    decade_file_start = (int(str(date)[:4]) // 10) * 10
    decade_file_end = min(decade_file_start + 9, year_max)
    dataz = xr.open_dataset(os.path.join(data_dir, f'ERA5_NH_{decade_file_start}-{decade_file_end}_Z500_Daily.nc'))
    print(f'Loaded Z500s: {decade_file_start}–{decade_file_end}')

    ### Load BLO data
    BLO_datasets = []
    for start_year in range(year_min, 2030, 10):
        end_year = min(start_year + 9, year_max)
        filename = os.path.join(data_dir, f'ERA5_BLO_1d_{start_year}-{end_year}_Daily.nc')
        if os.path.exists(filename):
            BLO_datasets.append(xr.open_dataset(filename))
            print(f'Loaded BLOs: {start_year}–{end_year}')
    BLOs = xr.concat(BLO_datasets, dim='time')
    print(f'Concatenated BLOs')

    ### Apply 5-day persistence, flagging every day of each run of at least five consecutive blocked days
    window_end = (BLOs.sortby('time').rolling(time=5, min_periods=5).sum() >= 5).astype(np.int8)  # 1 on the last day of each fully blocked 5-day window
    BLOs_5d = (window_end.rolling(time=5, min_periods=1).max().shift(time=-4, fill_value=0) >= 1).astype(np.int8)  # extend the flag back over the window

    ### Day specification
    a_day_dataz = dataz['z'].sel(time=date)
    a_day_blocking = BLOs_5d.sel(time=date)

    ### Figure design
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='w')
    plt.rc('xtick', color='k')
    plt.rc('ytick', color='k')
    plt.rc('axes', labelcolor='k')
    plt.rc('axes', facecolor='w')

    ### Create figure
    _, ax = plt.subplots(figsize=(6.5, 3.25), subplot_kw={'projection': ccrs.Robinson()})
    ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())
    ax.set_xticks([])
    ax.tick_params(axis='both', labelsize=5)
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontsize=5)
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='none', linewidth=.5, zorder=2); ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
    ax.gridlines(draw_labels=True, xlocs=[-140, -50, 40, 130], ylocs=[35, 75], linestyle='--', color='k')
    ax.gridlines(xlocs=[-140, -50, 40, 130], ylocs=[35, 75], linestyle='--', color='k')
    for x in [(-130, 'AM'), (-40, 'AT'), (55, 'AS'), (135, 'PA'), (-175, 'PA')]:
        ax.text(x[0], 2, x[1], transform=ccrs.PlateCarree(), fontsize=12, color='w')

    ### Plot geopotential heights
    a_day_dataz = ax.contourf(a_day_dataz.longitude, a_day_dataz.latitude, a_day_dataz,
                              levels=np.linspace(5000, 6000, 21), cmap=plt.get_cmap('RdBu_r', 192), transform=ccrs.PlateCarree(), extend='both')

    ### Plot blocked points
    lats, lons = a_day_blocking['BLO'].latitude, a_day_blocking['BLO'].longitude
    block_count = 0
    for lat in lats:
        for lon in lons:
            lat_idx, lon_idx = np.where(lats == lat)[0], np.where(lons == lon)[0]
            if a_day_blocking.BLO[lat_idx, lon_idx] == 1:
                block_count += 1
                ax.plot(lon, lat, marker='o', markersize=1, color='b', transform=ccrs.PlateCarree(), alpha=1, zorder=5)

    ### Colorbar
    cbar = plt.colorbar(a_day_dataz, orientation='horizontal', aspect=40, pad=.18, shrink=.8, extend='both')
    cbar.ax.tick_params(labelsize=10, colors='k', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
    cbar.ax.set_xticks(np.linspace(5000, 6000, 3)); cbar.ax.set_xticklabels(['5000', '5500', '6000'])
    cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.set_xlabel(r'\textbf{500 hPa Geopotential Height [gpm]}', fontsize=13, color='k', labelpad=8)

    ### Title
    plt.title(r'\textbf{Blocking Index (BLO)}', color='k', fontsize=20, y=1.22)

    ### Annotations
    plt.text(-.07, 1.32, str(a_day_blocking.time.values)[:10], transform=ax.transAxes, fontsize=17, verticalalignment='bottom', color='b')
    plt.text(-.07, 1.22, f'Blocked Points: {block_count}', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, color='b', fontsize=10)

    ### Save figure
    plt.subplots_adjust(top=.78, bottom=.12, left=.08, right=.92)
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_BLO_5d_{date}.png'), dpi=1000)
    plt.close()
    print()

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/NH/BLO/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Five Day Blocking Index (BLO) ----\n')

### Plot
plot_map('2010-01-06')
plot_map('1970-04-23'), plot_map('1970-11-10'), plot_map('1997-11-11'), plot_map('2000-05-02')  # birthdays

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')