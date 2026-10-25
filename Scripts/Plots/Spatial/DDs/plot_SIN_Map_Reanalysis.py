"""
Plot single day 500 hPa field and Sinuosity Index (SIN) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from skimage import measure
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

def plot_map(date):

    print(f'Plot {date}')

    ### Load Z500 data (single file only)
    decade_file_start = (int(str(date)[:4]) // 10) * 10
    decade_file_end = min(decade_file_start + 9, year_max)
    dataz = xr.open_dataset(os.path.join(data_dir, f'ERA5_NH_{decade_file_start}-{decade_file_end}_Z500_Daily.nc'))
    print(f'Loaded Z500s: {decade_file_start}–{decade_file_end}')

    ### Day specification
    a_day_dataz = dataz['z'].sel(time=date)

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
    ax.add_feature(cfeature.LAND, facecolor='none', linewidth=.5, zorder=2)
    ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)

    ### Gridlines and sectors
    ax.gridlines(draw_labels=True, xlocs=[-140, -50, 40, 130], ylocs=[30, 50, 70], linestyle='--', color='k')
    for x in [(-130, 'AM'), (-40, 'AT'), (55, 'AS'), (135, 'PA'), (-175, 'PA')]:
        ax.text(x[0], 2, x[1], transform=ccrs.PlateCarree(), fontsize=12, color='w')

    ### Plot
    cf = ax.contourf(a_day_dataz.longitude, a_day_dataz.latitude, a_day_dataz, levels=np.linspace(5000, 6000, 21), cmap=plt.get_cmap('RdBu_r', 192), transform=ccrs.PlateCarree(), extend='both')

    ### Colorbar
    cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.18, shrink=.8, extend='both')
    cbar.ax.tick_params(labelsize=10, colors='k', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
    cbar.ax.set_xticks(np.linspace(5000, 6000, 3))
    cbar.ax.set_xticklabels(['5000', '5500', '6000'])
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.set_xlabel(r'\textbf{500 hPa Geopotential Height [gpm]}', fontsize=13, color='k', labelpad=8)

    ### Isohype and sinuosity calculation and plotting
    mean_value_30_70N = cosine_weighted_spatial_mean(a_day_dataz.sel(latitude=slice(70, 30)).mean(dim='longitude'))
    cbar.ax.axvline(x=mean_value_30_70N.values, color='r', linestyle='-', linewidth=2)
    contour_value = mean_value_30_70N.values
    ax.contour(a_day_dataz.longitude, a_day_dataz.latitude, a_day_dataz, levels=[contour_value], colors='r', linewidths=2, linestyles='solid', transform=ccrs.PlateCarree(), zorder=3)
    contour_length = 0
    for contour in measure.find_contours(a_day_dataz.values, contour_value):
        contour_length += np.sum(np.sqrt((np.diff(contour[:, 1])) ** 2 + (np.diff(contour[:, 0])) ** 2))
    sinuosity = contour_length / len(a_day_dataz.longitude)
    print(f'Isohypse: {contour_value:.1f} | Sinuosity: {sinuosity:.2f}')

    ### Title
    plt.title(r'\textbf{Sinuosity Index (SIN)}', color='k', fontsize=18, y=1.22)

    ### Annotation
    plt.text(-.07, 1.33, str(a_day_dataz.time.values)[:10], transform=ax.transAxes, fontsize=17, verticalalignment='bottom', color='r')
    plt.text(-.07, 1.25, f'Isohypse: {mean_value_30_70N.values:,.1f} gpm', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, color='r', fontsize=8)
    plt.text(-.07, 1.1, f'Sinuosity Index: {sinuosity:.2f}', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, color='r', fontsize=8.8,
             bbox=dict(facecolor='w', edgecolor='r', boxstyle='round,pad=0.5'))

    ### Save figure
    plt.subplots_adjust(top=.78, bottom=.12, left=.08, right=.92)
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_SIN_{date}.png'), dpi=1000)
    plt.close()
    print()

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/NH/SIN/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Sinuosity Index (SIN) ----\n')

### Plot
plot_map('2010-01-06')
plot_map('1970-04-23'), plot_map('1970-11-10'), plot_map('1997-11-11'), plot_map('2000-05-02')  # birthdays

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')