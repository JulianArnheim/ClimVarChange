"""
Plot 100-m wind vectors (U100, V100) from ERA5 reanalysis

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
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Global/UV100/Historical/')

### Years
year_min, year_max = 1940, 2025

### Epoch
epoch = (1981, 2010)

print(f'\n---- 100-m Wind (U100, V100) ----\n')

### Load data
U_filename, V_filename = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_U100_Monthly.nc'), os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_V100_Monthly.nc')
datau, datav = xr.open_dataset(U_filename)['u100'], xr.open_dataset(V_filename)['v100']

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### Figure
    print(f'{text_months}')

    ### Process data
    U100, V100 = datau.sel(time=datau.time.dt.month.isin(mons)), datav.sel(time=datav.time.dt.month.isin(mons))
    if text_months == 'DJF':
        winter_year = xr.where(U100.time.dt.month == 12, U100.time.dt.year + 1, U100.time.dt.year)
        U100 = U100.assign_coords(winter_year=('time', winter_year.data)).groupby('winter_year').mean(dim='time').rename({'winter_year': 'year'})
        winter_year = xr.where(V100.time.dt.month == 12, V100.time.dt.year + 1, V100.time.dt.year)
        V100 = V100.assign_coords(winter_year=('time', winter_year.data)).groupby('winter_year').mean(dim='time').rename({'winter_year': 'year'})
    else:
        U100, V100 = U100.groupby('time.year').mean(dim='time'), V100.groupby('time.year').mean(dim='time')
    U100, V100 = U100.sel(year=slice(*epoch)).mean(dim='year'), V100.sel(year=slice(*epoch)).mean(dim='year')
    wind_mag = np.hypot(U100, V100)

    ### Spatial mean magnitude of the mean wind vector
    mean_val, min_wind_speed, max_wind_speed = cosine_weighted_spatial_mean(wind_mag.mean(dim='longitude')).compute().item(), wind_mag.min().compute().item(), wind_mag.max().compute().item()
    print(f'Global [Min/Mean/Max]: {min_wind_speed:.4f} / {mean_val:.4f} / {max_wind_speed:.4f} m s⁻¹')

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
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_xticks([])
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontsize=5)
    ax.coastlines(); ax.add_feature(cfeature.LAND, facecolor='none', linewidth=.5, zorder=2); ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
    gridlines = ax.gridlines(draw_labels=True, linewidth=.5, zorder=4)
    gridlines.xlabel_style = gridlines.ylabel_style = {'size': 7}

    ### Plot
    x, y = np.meshgrid(wind_mag.longitude, wind_mag.latitude)
    cf = ax.contourf(U100.longitude, U100.latitude, wind_mag, levels=np.arange(0, 15.1, 1), cmap=plt.get_cmap('YlOrRd', lut=192), transform=ccrs.PlateCarree(), extend='max')
    ax.quiver(U100.longitude.values[::50], U100.latitude.values[::50], U100.values[::50, ::50], V100.values[::50, ::50], transform=ccrs.PlateCarree(), scale=300, width=.002, zorder=5)

    ### Colorbar
    cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.1, shrink=.8, extend='both')
    cbar.ax.tick_params(labelsize=10, color='dimgray', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
    cbar.ax.set_xticks(np.linspace(0, 15, 4)); cbar.ax.set_xticklabels(['0', '5', '10', '15'])
    cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.set_xlabel(r'\textbf{Magnitude of Mean Wind [m s\textsuperscript{-1}]}', fontsize=13, color='k', labelpad=8)

    ### Title
    plt.title(r'\textbf{100-m WIND}', color='k', fontsize=20, y=1.12)

    ### Annotation
    plt.text(.88, 1.05, rf'\textbf{{{"" if text_months == "ANNUAL" else f"{text_months} "}{epoch[0]}–{epoch[1]}}}', fontsize=9, rotation='horizontal', ha='left', va='top', transform=ax.transAxes)

    ### Save figure
    plt.subplots_adjust(top=.98, bottom=.03, left=.08, right=.92)
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_Global_UV100_{text_months}_{epoch[0]}–{epoch[1]}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')