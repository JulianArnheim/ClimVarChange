"""
Plot storm-track activity from Blackmon (1976), Blackmon et al. (1977), Iversen et al. (2023) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from dask.diagnostics import ProgressBar
import cartopy.crs as ccrs
import cmocean
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np
import os
import time
import xarray as xr

mpl.use('Agg')  # non-interactive backend to avoid tkinter issues

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

def lanczos_bandpass_filter(data, low_days=6, high_days=2, window=121, fs=1.0):
    """Default: 2-6 day bandpass (2 days is the Nyquist period of daily data), 121-point filter window (60 days each side)."""
    M = (window - 1) // 2
    n = np.arange(-M, M + 1)  # index array centered at zero
    cl, ch = (1 / low_days), (1 / high_days)  # cutoffs in cycles/day
    h_low = 2 * cl / fs * np.sinc(2 * cl * n / fs)  # low cutoff filter (highpass) sinc function
    h_high = 2 * ch / fs * np.sinc(2 * ch * n / fs)  # high cutoff filter (lowpass) sinc function
    h = (h_high - h_low) * np.sinc(n / M)  # bandpass kernel times a lanczos filter to dampen side lobes, reducing Gibbs ringing
    conv_func = lambda arr: np.convolve(arr, h, mode='same')
    ufunc_kwargs = dict(input_core_dims=[['time']], output_core_dims=[['time']])
    apply_kwargs = dict(vectorize=True, dask='parallelized', output_dtypes=[data.dtype])
    filtered = xr.apply_ufunc(conv_func, data, **ufunc_kwargs, **apply_kwargs).isel(time=slice(M, -M))  # drop edge points contaminated by truncation
    return filtered

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/NA/Stormtrack/')

### Years and epochs
year_min, year_max = 1940, 2025
epochs = [(year_min, year_max), (1981, 2010)]

print(f'\n---- Storm-Track Activity ----\n')

### Loop through epochs
for start_year, end_year in epochs:

    ### First valid year
    first_year = start_year + 1 if start_year == year_min else start_year

    ### Load data
    dataz_datasets = []
    for dec_start in range((start_year // 10) * 10, (end_year // 10) * 10 + 1, 10):
        dec_end = dec_start + 9
        if dec_end > year_max:
            dec_end = year_max
        filename = os.path.join(data_dir, f'ERA5_NH_{dec_start}-{dec_end}_Z500_Daily.nc')
        if os.path.exists(filename):
            ds_region = xr.open_dataset(filename, chunks={'time': 30})['z'].sel(latitude=slice(70, 25), longitude=slice(-60, 0))
            dataz_datasets.append(ds_region)
            print(f'Loaded: {dec_start}–{dec_end}')
        else:
            print(f'File not found: {dec_start}–{dec_end}')
    dataz_region = xr.concat(dataz_datasets, dim='time')
    print()

    ### Time specification with at least 60 days of padding on each side for the filter edges
    dataz_time = dataz_region.sel(time=slice(f'{first_year-1}-10-01', f'{end_year}-04-30'))

    ### Loop through filtering
    for apply_filter in [False, True]:

        if apply_filter:
            tag, dataz_filter = 'bandpass', lanczos_bandpass_filter(dataz_time.chunk(dict(time=-1)))

        else:
            tag, dataz_filter = 'raw', dataz_time

        print(f'DJF | {first_year}–{end_year} ({tag})')

        ### Season specification
        dataz_season = dataz_filter.sel(time=dataz_filter.time.dt.month.isin([12, 1, 2]))
        month, year = dataz_season.time.dt.month, dataz_season.time.dt.year
        winter_year = xr.where(month == 12, year + 1, year)
        dataz_season = dataz_season.assign_coords(winter_year=('time', winter_year.data)).where((winter_year >= first_year) & (winter_year <= end_year), drop=True)

        ### Storm-track activity as standard deviation
        with ProgressBar():
            std_map = dataz_season.std(dim='time', skipna=True).compute()

        ### Figure design
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
        plt.rc('savefig', facecolor='w')
        plt.rc('xtick', color='k')
        plt.rc('ytick', color='k')
        plt.rc('axes', labelcolor='k')
        plt.rc('axes', facecolor='w')
        plt.rc('xtick', labelsize=11)
        plt.rc('ytick', labelsize=11)

        ### Create figure
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(), position=[.1, 0, .8, .75])
        ax.coastlines(linewidth=1, resolution='50m')
        ax.gridlines(draw_labels=True, linewidth=.5, zorder=6)
        ax.set_extent([-60, 0, 25, 70], crs=ccrs.PlateCarree())

        ### Plot
        x, y = np.meshgrid(std_map.longitude, std_map.latitude)
        cf = ax.contourf(x, y, std_map, levels=np.arange(0, 201, 10), cmap=cmocean.cm.dense, transform=ccrs.PlateCarree(), extend='max')

        ### Colorbar
        ticks = np.arange(0, 201, 50)
        cbar = plt.colorbar(cf, orientation='horizontal', aspect=30, pad=.09, shrink=.9, ticks=ticks, extend='max')
        cbar.set_label(r'\textbf{Standard Deviation of 500 hPa Geopotential Height [gpm]}', fontsize=11, labelpad=9, color='k', fontfamily='Avant Garde')
        cbar.set_ticklabels([f'{int(i)}' for i in ticks])

        ### Title
        plt.title(r'\textbf{Storm-Track Activity}', color='k', fontsize=18, y=1.09)\

        ### Annotation
        plt.text(1.1, 1.12, rf'\textbf{{DJF {first_year}–{end_year}}}', fontsize=10, ha='right', va='top', transform=ax.transAxes)

        ### Maximum value and location
        max_index = np.unravel_index(std_map.values.argmax(), std_map.values.shape)
        max_lat_epoch, max_lon_epoch = std_map.latitude.values[max_index[0]], std_map.longitude.values[max_index[1]]
        print(f'Maximum: {std_map.values.max():.2f} gpm at {max_lat_epoch:.2f}°N, {abs(max_lon_epoch):.2f}°W')
        ax.plot(max_lon_epoch, max_lat_epoch, marker='*', color='gold', markersize=8, transform=ccrs.PlateCarree(), zorder=8)

        ### Save figure
        plt.subplots_adjust(bottom=.01)
        fig_path = os.path.join(fig_dir, f'Reanalysis_NA_Z500_Stormtrack_{tag}_DJF_{first_year}–{end_year}.png')
        plt.savefig(fig_path, dpi=1000)
        plt.close(fig)

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')