"""
Plot Sea Surface Temperature (SST) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.ndimage import binary_dilation, label
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

def patch_small_nan_clusters(data, max_size=20, min_perimeter_fraction=2/3, dilation_iters=2):
    arr = data.values.copy()
    labels, nlabels = label(np.isnan(arr))
    for i in range(1, nlabels + 1):
        cluster = labels == i
        if cluster.sum() > max_size:
            continue
        ring = binary_dilation(cluster, iterations=dilation_iters)
        ring &= ~cluster
        total_ring = ring.sum()
        if total_ring == 0:
            continue
        valid_ring = np.isfinite(arr[ring])
        if valid_ring.sum() / total_ring < min_perimeter_fraction:
            continue
        values = arr[ring][valid_ring]
        arr[cluster] = np.median(values)
    return xr.DataArray(arr, coords=data.coords, dims=data.dims, attrs=data.attrs)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Global/SST/Historical/')

### Years and epoch
year_min, year_max, epoch = 1940, 2025, (1981, 2010)

print(f'\n---- Sea Surface Temperature (SST) ----\n')

### Load data
filename = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_SST_Monthly.nc')
datat = xr.open_dataset(filename)['sst']

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### Figure
    print(f'{text_months}')

    ### Process data
    SST = datat.sel(time=datat['time'].dt.month.isin(mons))
    if text_months == 'DJF':
        month, year = SST.time.dt.month, SST.time.dt.year
        winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
        SST = SST.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
        SST = SST.groupby('winter_year').mean(dim='time').rename({'winter_year': 'year'})
    else:
        SST = SST.groupby('time.year').mean(dim='time')
    mean_map = SST.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')

    ### Print spatial mean statistics
    mean_val = (cosine_weighted_spatial_mean(mean_map.sum(dim='longitude')) / cosine_weighted_spatial_mean(mean_map.notnull().sum(dim='longitude'))).compute().item()  # 2-D area mean over ocean points
    min_temperature, max_temperature = mean_map.min().compute().item(), mean_map.max().compute().item()
    print(f'Global [Min/Mean/Max]: {min_temperature:.4f} / {mean_val:.4f} / {max_temperature:.4f} °C')

    ### Patch islands after the statistics (display only)
    mean_map = patch_small_nan_clusters(mean_map)

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
    ax.coastlines()
    ax.add_feature(cfeature.LAND, facecolor='dimgray', linewidth=.5, zorder=2); ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
    gridlines = ax.gridlines(draw_labels=True, linewidth=.5, zorder=4)
    gridlines.xlabel_style = gridlines.ylabel_style = {'size': 7}

    ### Plot
    x = np.linspace(0, 1, 256)
    x[x <= .5] = .5 - .8 * (.5 - x[x <= .5])
    cmap_SST = LinearSegmentedColormap.from_list('RdBu_r_SST', plt.get_cmap('RdBu_r')(x))
    x, y = np.meshgrid(datat.longitude, datat.latitude)
    cf = ax.contourf(x, y, mean_map, levels=np.arange(-10, 42, 2), cmap=cmap_SST, vmin=-10, vmax=40, norm=TwoSlopeNorm(vmin=-10, vcenter=0, vmax=40), extend='max', transform=ccrs.PlateCarree())

    ### Colorbar
    cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.1, shrink=.8, extend='both')
    cbar.ax.tick_params(labelsize=10, color='dimgray', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
    cbar.ax.set_xticks(np.linspace(-10, 40, 6)); cbar.ax.set_xticklabels(['-10', '0', '10', '20', '30', '40'])
    cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.set_xlabel(r'\textbf{Sea Surface Temperature [°C]}', fontsize=13, color='k', labelpad=8)

    ### Title
    plt.title(r'\textbf{SEA SURFACE TEMPERATURE}', color='k', fontsize=20, y=1.12)

    ### Annotations
    plt.text(.88, 1.05, rf'\textbf{{{"" if text_months == "ANNUAL" else f"{text_months} "}{epoch[0]}–{epoch[1]}}}', fontsize=9, rotation='horizontal', ha='left', va='top', transform=ax.transAxes)
    plt.text(.12, 1.07, rf'\textbf{{{mean_val:.2f}°C}}', fontsize=9, ha='right', va='top', transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='k', boxstyle='round,pad=0.2'))

    ### Save figure
    plt.subplots_adjust(top=.98, bottom=.03, left=.08, right=.92)
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_Global_SST_{text_months}_{epoch[0]}–{epoch[1]}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')