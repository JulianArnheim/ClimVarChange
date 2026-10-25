"""
Plot Sea Ice Concentration (ICEFRAC) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from cartopy.util import add_cyclic_point
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs, cartopy.feature as cfeature
import cmocean
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
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Arctic/ICEFRAC/Historical/')

### Years and epoch
year_min, year_max, epoch = 1940, 2025, (1981, 2010)

print(f'\n---- Sea Ice Concentration (ICEFRAC) ----\n')

### Load data
filename = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_ICEFRAC_Monthly.nc')
datac = xr.open_dataset(filename)['siconc']

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### Figure
    print(f'{text_months}')

    ### Process data
    ICEFRAC = datac.sel(time=datac['time'].dt.month.isin(mons))
    ICEFRAC = ICEFRAC.where(ICEFRAC.latitude >= 30, drop=True) * 100
    if text_months == 'DJF':
        month, year = ICEFRAC.time.dt.month, ICEFRAC.time.dt.year
        winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
        ICEFRAC = ICEFRAC.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
        ICEFRAC = ICEFRAC.groupby('winter_year').mean(dim='time').rename({'winter_year': 'year'})
    else:
        ICEFRAC = ICEFRAC.groupby('time.year').mean(dim='time')
    mean_map = ICEFRAC.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')

    ### Arctic sea ice area and extent for an oblate spheroid Earth
    R_eq, R_polar = 6378137.0, 6356752.3  # equatorial and polar radii [m]
    e2 = 1 - (R_polar**2 / R_eq**2)  # eccentricity^2
    lat_rad, lon_rad = np.deg2rad(mean_map.latitude.values), np.deg2rad(mean_map.longitude.values)
    dlat, dlon = np.abs(np.diff(lat_rad).mean()), np.abs(np.diff(lon_rad).mean())
    Rm = R_eq * (1 - e2) / (1 - e2 * np.sin(lat_rad) ** 2) ** 1.5  # meridional radius of curvature
    Rp = R_eq / np.sqrt(1 - e2 * np.sin(lat_rad) ** 2)  # prime vertical radius of curvature
    cell_area_obl = Rm * Rp * np.cos(lat_rad) * dlat * dlon
    cell_area_obl = np.broadcast_to(cell_area_obl[:, np.newaxis], (len(lat_rad), len(lon_rad)))
    cell_area_obl = xr.DataArray(cell_area_obl, coords=mean_map.coords, dims=mean_map.dims)
    ice_area_obl_mkm2 = (cell_area_obl * mean_map / 100).where(mean_map).sum(dim=['latitude', 'longitude']) / 1e12
    ice_extent_obl_mkm2 = cell_area_obl.where(mean_map >= 15).sum(dim=['latitude', 'longitude']) / 1e12
    print(f'Arctic sea ice area: {ice_area_obl_mkm2:.2f} million km²')
    print(f'Arctic sea ice extent: {ice_extent_obl_mkm2:.2f} million km²')

    ### Extrapolate NaNs
    mean_map = mean_map.sortby('latitude', 'longitude')
    mean_map = mean_map.interpolate_na(dim='latitude', method='nearest', fill_value='extrapolate')
    mean_map = mean_map.interpolate_na(dim='longitude', method='nearest', fill_value='extrapolate')

    ### Figure design
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='w')
    plt.rc('axes', edgecolor='k')
    plt.rc('xtick', color='dimgray')
    plt.rc('ytick', color='dimgray')
    plt.rc('axes', labelcolor='dimgray')
    plt.rc('axes', facecolor='w')

    ### Create figure
    fig = plt.figure(figsize=(5.5, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
    ax.set_xticks([])
    ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontsize=5)

    ### Plot
    data_cyclic, lon_cyclic = add_cyclic_point(mean_map.values, coord=mean_map.longitude)
    x_wrap, y_wrap = np.meshgrid(lon_cyclic, mean_map.latitude.values)
    cf = ax.contourf(x_wrap, y_wrap, data_cyclic, levels=np.arange(0, 100.1, 5), cmap=cmocean.cm.ice, vmin=0, vmax=100, transform=ccrs.PlateCarree())
    cs_edge = ax.contour(x_wrap, y_wrap, data_cyclic, levels=[15], colors='magenta', linewidths=1, transform=ccrs.PlateCarree(), zorder=1)

    ### Features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', linewidth=.5, zorder=1)
    ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
    ax.add_feature(cfeature.RIVERS.with_scale('50m'), edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
    ax.coastlines(linewidth=.5, resolution='50m', zorder=3)

    ### Gridlines
    for lon in np.arange(-180, 181, 60):
        ax.plot([lon, lon], [0, 90], transform=ccrs.PlateCarree(), color='gray', linewidth=.5, zorder=5)
    for lat in np.arange(0, 91, 10):
        ax.plot(np.linspace(-180, 180, 361), np.full(361, lat), transform=ccrs.PlateCarree(), color='gray', linewidth=.5, zorder=5)

    ### Colorbar
    cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.06, shrink=.8)
    cbar.ax.tick_params(labelsize=10, colors='k', labelright=True, labelleft=False, right=True, left=False)
    cbar.ax.set_xticks(np.linspace(0, 100, 5)); cbar.ax.set_xticklabels(['0', '25', '50', '75', '100'])
    cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.set_xlabel(r'\textbf{Sea Ice Concentration [\%]}', fontsize=13, color='k', labelpad=8)
    cbar.ax.axvline(15, color='magenta', linestyle='-', linewidth=1.5)
    cbar.outline.set_edgecolor('k')
    cbar.outline.set_linewidth(.8)

    ### Title and annotations
    plt.title(r'\textbf{SEA ICE CONCENTRATION}', color='k', fontsize=20, y=1.05)
    season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
    plt.text(1, 1.04, rf'\textbf{{{season_prefix}{epoch[0]}–{epoch[1]}}}', fontsize=9, rotation='horizontal', ha='right', va='top', transform=ax.transAxes)
    annotation = rf'\raggedright\textbf{{Area (Extent): {ice_area_obl_mkm2:.1f} ({ice_extent_obl_mkm2:.1f}) million km\textsuperscript{{2}}}}'
    plt.text(.02, .98, annotation, fontsize=7, ha='left', va='top', transform=ax.transAxes, bbox=dict(facecolor='w', edgecolor='k', boxstyle='round,pad=0.35'), zorder=6)

    ### Ensure outline is on top
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none', edgecolor='k', linewidth=1.2, zorder=10))

    ### Save figure
    plt.subplots_adjust(top=.88, bottom=.02, left=.02, right=.98)
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_Arctic_ICEFRAC_{text_months}_{epoch[0]}–{epoch[1]}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')