"""
Plot the interannual covariance of seasonal-mean 850 hPa meridional wind and temperature, after Liang et al. (2025), from
ERA5 reanalysis (monthly data cannot resolve the transient eddy heat flux, which needs daily or 6-hourly band-passed data)

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from cartopy.util import add_cyclic_point
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
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/NA/Baroclinicity/')

### Years
year_min, year_max = 1940, 2025

### Epochs
epochs = [(year_min, year_max), (1981, 2010)]

print(f'\n---- Interannual Covariance of Seasonal-Mean v850 and T850 ----\n')

### Load T data
datat = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    T_filename = os.path.join(data_dir, f'ERA5_Global_{start_year}-{end_year}_TProfile_Monthly.nc')
    if os.path.exists(T_filename):
        ds = xr.open_dataset(T_filename).sel(level=850, latitude=slice(90, 20)).astype(np.float32)
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
        ds = xr.open_dataset(V_filename).sel(level=850, latitude=slice(90, 20)).astype(np.float32)
        datav.append(ds)
        print(f'Loaded V: {start_year}–{end_year}')
if datav:
    datav = xr.concat(datav, dim='time').sortby('time')
else:
    raise FileNotFoundError('No ERA5_Global_*_VProfile_Monthly.nc files found.')
print()

### Loop through epochs
for one_start_year, one_end_year in epochs:

    ### Month settings
    months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

    ### Loop through each set of months
    for sel_month, mons in months_options.items():

        ### String corrections
        text_months = sel_month.split()[1]

        ### First valid year
        first_year = one_start_year + 1 if text_months == 'DJF' and one_start_year == 1940 else one_start_year

        ### Figure
        print(f'{text_months} | {one_start_year}–{one_end_year}')

        ### Process data
        T, V = datat['t'].sel(latitude=slice(90, 20)) + 273.15, datav['v'].sel(latitude=slice(90, 20))
        if text_months == 'DJF':
            T, V = T.sel(time=T.time.dt.month.isin(mons)), V.sel(time=V.time.dt.month.isin(mons))
            winter_year = xr.where(T.time.dt.month == 12, T.time.dt.year + 1, T.time.dt.year)  # T and V share the same monthly time axis
            T, V = (da.assign_coords(winter_year=('time', winter_year.data)).groupby('winter_year').mean(dim='time')
                    .rename({'winter_year': 'year'}).sel(year=slice(first_year, one_end_year)) for da in (T, V))
        else:
            T = T.sel(time=T.time.dt.month.isin(mons)).groupby('time.year').mean(dim='time').sel(year=slice(first_year, one_end_year))
            V = V.sel(time=V.time.dt.month.isin(mons)).groupby('time.year').mean(dim='time').sel(year=slice(first_year, one_end_year))
        T_mean, V_mean = T.mean(dim='year'), V.mean(dim='year')

        ### Interannual covariance of the seasonal means (not the transient eddy heat flux)
        T_prime, V_prime = T - T_mean, V - V_mean  # deviations of each seasonal mean from the multi-year mean
        heat_flux = (T_prime * V_prime).mean(dim='year')  # covariance of seasonal-mean v and T

        ### Figure design
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
        plt.rc('savefig', facecolor='w')
        plt.rc('axes', edgecolor='k')
        plt.rc('xtick', color='k')
        plt.rc('ytick', color='k')
        plt.rc('axes', labelcolor='k', facecolor='w')

        ### Create figure
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(), position=[.1, 0, .8, .75])
        ax.coastlines(linewidth=1, resolution='50m')
        ax.gridlines(draw_labels=True, linewidth=.5, zorder=6)
        ax.set_extent([-60, 0, 25, 70], crs=ccrs.PlateCarree())

        ### Plot
        cmap = plt.get_cmap('RdBu_r', lut=192)
        data_cyclic, lon_cyclic = add_cyclic_point(heat_flux.values, coord=heat_flux.longitude)
        x_wrap, y_wrap = np.meshgrid(lon_cyclic, heat_flux.latitude.values)
        cf = ax.contourf(x_wrap, y_wrap, data_cyclic, levels=np.arange(-10, 10.1, 1), cmap=cmap, transform=ccrs.PlateCarree(), extend='both')

        ### Colorbar
        ticks = np.arange(-10, 10.1, 5)
        cbar = plt.colorbar(cf, orientation='horizontal', aspect=30, pad=.09, shrink=.9, ticks=ticks, extend='both')
        cbar.ax.tick_params(labelsize=10, color='k', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
        cbar.set_ticklabels([f'{i:.1f}'.lstrip('0').replace('0.', '.') if 0 < i < 1 else f'{int(i)}' for i in ticks])
        cbar.ax.xaxis.set_ticks_position('bottom')
        cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.set_xlabel(r'\textbf{Interannual Covariance of Seasonal-Mean $v$ and $T$ [K m s\textsuperscript{{-1}}]}', fontsize=11, color='k', labelpad=8)

        ### Title and annotation
        plt.title(r'$\overline{[v]^\prime [T]^\prime}$', fontsize=22, color='k', y=1.08)
        plt.text(1.1, 1.12, rf"\textbf{{{'' if text_months == 'ANNUAL' else f'{text_months} '}{first_year}–{one_end_year}}}", fontsize=10, ha='right', va='top', transform=ax.transAxes)

        ### Save figure
        plt.subplots_adjust(bottom=.01)
        plt.savefig(os.path.join(fig_dir, f'Reanalysis_Interannual_vT_Covariance_{text_months}_{first_year}–{one_end_year}.png'), dpi=1000)
        plt.close()

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')