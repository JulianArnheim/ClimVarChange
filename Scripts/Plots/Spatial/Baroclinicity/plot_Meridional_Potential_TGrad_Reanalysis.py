"""
Plot meridional potential temperature gradient from Liang et al. (2025) from ERA5 reanalysis

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

### Years and epochs
year_min, year_max, epochs = 1940, 2025, [(1940, 2025), (1981, 2010)]

print(f'\n---- Meridional Potential Temperature Gradient ----\n')

### Load data
datat = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_Global_{start_year}-{end_year}_TProfile_Monthly.nc')
    if os.path.exists(filename):
        datat.append(xr.open_dataset(filename, chunks={'time': 12, 'latitude': 16, 'level': 10}).astype(np.float32))
        print(f'Loaded: {start_year}–{end_year}')
if datat:
    datat = xr.concat(datat, dim='time').sortby('time')
else:
    raise FileNotFoundError('No ERA5_Global_*_TProfile_Monthly.nc files found.')
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
        T850 = datat['t'].sel(level=850, latitude=slice(90, 20))
        if text_months == 'DJF':
            da = T850.sel(time=T850.time.dt.month.isin(mons))
            winter_year = xr.where(da.time.dt.month == 12, da.time.dt.year + 1, da.time.dt.year)
            T850 = da.assign_coords(winter_year=('time', winter_year.data)).groupby('winter_year').mean(dim='time').rename({'winter_year': 'year'}).sel(year=slice(first_year, one_end_year))
        else:
            T850 = T850.sel(time=T850.time.dt.month.isin(mons)).groupby('time.year').mean(dim='time').sel(year=slice(first_year, one_end_year))
        theta850 = (T850 * (1000 / 850) ** .286).mean(dim='year').chunk({'latitude': -1})

        ### Gradient calculation assuming WGS84 ellipsoid
        R_eq, R_polar = 6378137.0, 6356752.3  # equatorial and polar radii [m]
        e2 = 1 - (R_polar**2 / R_eq**2)  # first eccentricity squared
        Rm = R_eq * (1 - e2) / (1 - e2 * np.sin(np.deg2rad(theta850.latitude)) ** 2) ** 1.5  # meridional radius of curvature [m]
        dy_dlat = xr.DataArray(Rm * np.pi / 180, coords={'latitude': theta850.latitude}, dims='latitude')  # meridional arc length per degree of latitude [m]
        theta850_gradient = -1e6 * theta850.differentiate('latitude') / dy_dlat

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
        ax.coastlines(linewidth=1, resolution='50m'); ax.gridlines(draw_labels=True, linewidth=.5, zorder=6); ax.set_extent([-60, 0, 25, 70], crs=ccrs.PlateCarree())

        ### Plot
        data_cyclic, lon_cyclic = add_cyclic_point(theta850_gradient.values, coord=theta850_gradient.longitude)
        x_wrap, y_wrap = np.meshgrid(lon_cyclic, theta850_gradient.latitude.values)
        cf = ax.contourf(x_wrap, y_wrap, data_cyclic, levels=np.arange(-20, 20.1, 2), cmap=plt.get_cmap('RdBu_r', lut=192), transform=ccrs.PlateCarree(), extend='both')

        ### Colorbar
        cbar = plt.colorbar(cf, orientation='horizontal', aspect=30, pad=.09, shrink=.9, ticks=np.arange(-20, 20.1, 10), extend='both')
        cbar.ax.tick_params(labelsize=10, color='k', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
        cbar.set_ticklabels([f'{int(i)}' for i in np.arange(-20, 20.1, 10)]); cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.set_xlabel(r'\textbf{850 hPa Meridional Potential Temperature Gradient [10\textsuperscript{{-6}} K m\textsuperscript{{-1}}]}', fontsize=11, color='k', labelpad=8)

        ### Title and annotation
        plt.title(r'$-\partial \bar{\theta}_{850}/\partial y$', color='k', fontsize=20, y=1.1)
        plt.text(1.1, 1.12, rf"\textbf{{{'' if text_months == 'ANNUAL' else f'{text_months} '}{first_year}–{one_end_year}}}", fontsize=10, ha='right', va='top', transform=ax.transAxes)

        ### Save figure
        plt.subplots_adjust(bottom=.01)
        plt.savefig(os.path.join(fig_dir, f'Reanalysis_Meridional_Potential_TGrad_{text_months}_{first_year}–{one_end_year}.png'), dpi=1000)
        plt.close()

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')