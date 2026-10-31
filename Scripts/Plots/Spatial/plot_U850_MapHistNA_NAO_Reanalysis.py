"""
Plot zonal wind at 850 hPa (U850) composited on the third of DJF days with the highest and lowest NAO index,
using the NOAA PSL daily NAO index (NCEP/NCAR R1 500 hPa heights truncated to total wavenumber 10, 1981–2010
anomalies, 35–45°N minus 55–70°N means over 70–10°W), from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Data : https://psl.noaa.gov/data/timeseries/daily/NAO/
    Author : Julian B Arnheim
"""

### Import modules
import cartopy.crs as ccrs
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np
import os
import time
import xarray as xr

mpl.use('Agg')  # non-interactive backend to avoid tkinter issues

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/NA/NAO/')

### Years and epochs
year_min, year_max = 1940, 2025
epochs = [(1948, year_max), (1981, 2010)]

### Pressure level
lev = 850

print(f'\n---- North Atlantic Oscillation (NAO) ----\n')

### Load zonal wind data
datau_datasets = []
for start_year in range(year_min, year_max + 1, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_NA_{start_year}-{end_year}_U{lev}_Daily.nc')
    if os.path.exists(filename):
        ds = xr.open_dataset(filename)['u']
        ds_region = ds.sel(latitude=slice(70, 25), longitude=slice(-60, 0))
        datau_datasets.append(ds_region)
        print(f'Loaded: {start_year}–{end_year}')
    else:
        print(f'File not found: {start_year}–{end_year}')
datau_region = xr.concat(datau_datasets, dim='time')
print()

### Load NAO data
datap_NAO = xr.open_dataarray(os.path.join(data_dir, f'PSL_NAO_1948-{year_max}_Daily.nc'))

### Loop through epochs
for start_yr, end_yr in epochs:

    print(f'DJF | {start_yr}–{end_yr}')

    ### Time specification of winds (U)
    datau_time = datau_region.sel(time=slice(f'{start_yr-1}-12-01', f'{end_yr}-02'))

    ### Time specification of NAO
    NAO_time = datap_NAO.sel(time=slice(f'{start_yr-1}-12-01', f'{end_yr}-02'))

    ### Season specification of winds (U)
    datau_season = datau_time.sel(time=datau_time.time.dt.month.isin([12, 1, 2]))
    month, year = datau_season.time.dt.month, datau_season.time.dt.year
    winter_year = xr.where(month == 12, year + 1, year)
    datau_season = datau_season.assign_coords(winter_year=('time', winter_year.data))
    datau_season = datau_season.sel(time=datau_season.winter_year.isin(range(start_yr, end_yr + 1)))

    ### Season specification of NAO
    NAO_season = NAO_time.where(NAO_time.time.dt.month.isin([12, 1, 2]), drop=True)
    month, year = NAO_season.time.dt.month, NAO_season.time.dt.year
    winter_year = xr.where(month == 12, year + 1, year)
    NAO_season = NAO_season.assign_coords(winter_year=('time', winter_year.data))
    NAO_season = NAO_season.where(NAO_season.winter_year.isin(range(start_yr, end_yr + 1)), drop=True)

    ### Rank NAO days
    all_days = NAO_season.sizes['time']
    extreme_days = all_days // 3
    print(f'Extreme third of days ({extreme_days} of {all_days})')
    NAO_sorted = NAO_season.sortby(NAO_season)
    NAO_highest, NAO_lowest = NAO_sorted[-extreme_days:].sortby('time'), NAO_sorted[:extreme_days].sortby('time')

    ### Loop through phases
    for phase in ['NAO+', 'NAO-', 'All']:

        ### Select by day
        phase_dates = {'NAO+': NAO_highest.time.values, 'NAO-': NAO_lowest.time.values, 'All': datau_season.time.values}
        mean_map = datau_season.sel(time=phase_dates[phase]).mean(dim='time').compute()

        ### Figure design
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
        plt.rc('savefig', facecolor='w')
        plt.rc('xtick', color='k')
        plt.rc('ytick', color='k')
        plt.rc('axes', facecolor='w', labelcolor='k')
        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)

        ### Create figure
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(), position=[.1, 0, .8, .75])
        ax.coastlines(linewidth=1, resolution='50m')
        ax.gridlines(draw_labels=True, linewidth=.5, zorder=6)
        ax.set_extent([-60, 0, 25, 70], crs=ccrs.PlateCarree())

        ### Plot
        x, y = np.meshgrid(mean_map.longitude, mean_map.latitude)
        cf = ax.contourf(x, y, mean_map, levels=np.arange(-20, 20.01, 2), cmap=plt.get_cmap('PuOr', lut=192), transform=ccrs.PlateCarree(), extend='both')

        ### Contours
        positive_contours = ax.contour(x, y, mean_map, levels=[12, 14, 16], colors='lightgray', linestyles='solid', linewidths=1, transform=ccrs.PlateCarree(), extend='both')
        plt.clabel(positive_contours, fmt='%d', colors='w', fontsize=9, inline=True, zorder=10)
        zero_contour = ax.contour(x, y, mean_map, levels=[0], colors='k', linestyles='solid', linewidths=1, transform=ccrs.PlateCarree(), extend='both')
        plt.clabel(zero_contour, fmt='%d', colors='k', fontsize=9, inline=True, zorder=10)

        ### Colorbar
        ticks = np.arange(-20, 20.1, 10)
        cbar = plt.colorbar(cf, orientation='horizontal', aspect=30, pad=.09, shrink=.9, ticks=ticks, extend='both')
        cbar.set_label(rf'\textbf{{Zonal Wind at {lev} hPa [m s\textsuperscript{{-1}}]}}', fontsize=14, labelpad=9, color='k', fontfamily='Avant Garde')
        cbar.set_ticklabels([f'{int(i)}' for i in ticks])

        ### Title
        plt.title(rf'\textbf{{$\bar{{U}}_{{{lev}}}$ on {phase} Days}}', color='k', fontsize=18, y=1.1)

        ### Annotation
        plt.text(1.1, 1.12, rf'\textbf{{DJF {start_yr}–{end_yr}}}', fontsize=10, rotation='horizontal', ha='right', va='top', transform=ax.transAxes)

        ### Save figure
        plt.subplots_adjust(bottom=.01)
        fig_path = os.path.join(fig_dir, f'Reanalysis_U{lev}_NA_{phase}_DJF_{start_yr}–{end_yr}.png')
        plt.savefig(fig_path, dpi=1000)
        plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')