"""
Plot Arctic and global mean Precipitation (PRECT) anomalies, "Arctic Amplification," from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import matplotlib as mpl, matplotlib.pyplot as plt, matplotlib.ticker as ticker
import numpy as np
import os
import time
import xarray as xr

mpl.use('Agg')  # non-interactive backend to avoid tkinter issues

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks_position('bottom')

def cosine_weighted_spatial_mean(data, coord=None):
    coord = coord or ('lat' if 'lat' in data.coords else 'latitude' if 'latitude' in data.coords else None)
    if not coord:
        raise ValueError("No 'lat' or 'latitude' coordinate found in the data array.")
    if 'lat' in data.coords and 'latitude' in data.coords:
        raise ValueError("Both 'lat' and 'latitude' coordinates exist.")
    weights = np.cos(np.deg2rad(data[coord]))
    return (data * weights).sum(dim=coord) / (data.notnull() * weights).sum(dim=coord)

def custom_formatter(x, _):
    return '0' if abs(x) < .01 else f'{x:+.1f}'.replace('+0', '+').replace('-0', '-').replace('.0', '')

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/PRECT/Arctic/')

### Years and epoch
year_min, year_max = 1940, 2025
epoch = (1981, 2010)

### Arctic definition
Arctic_min_lats = [60, 65, 66.5, 70]

print(f'\n---- Arctic Amplification (PRECT) ----\n')

### Load monthly mean PRECC and PRECL
file_precc_monthly = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_PRECC_Monthly.nc')
datap_precc = xr.open_dataset(file_precc_monthly, chunks={'time': 120})
file_precl_monthly = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_PRECL_Monthly.nc')
datap_precl = xr.open_dataset(file_precl_monthly, chunks={'time': 120})

### Calculate annual mean PRECT
datap_monthly = xr.Dataset()
datap_monthly['PRECT'] = datap_precc['PRECC'] + datap_precl['PRECL']
datap_annual = datap_monthly.groupby('time.year').sum(dim='time')

### Loop through each Arctic definition
for Arctic_min_lat in Arctic_min_lats:

    print(f'Arctic Minimum Latitude: {Arctic_min_lat}°N')

    ### Process data
    annual_zonal_mean_datap = datap_annual.sortby('year').mean(dim='longitude')

    ### Arctic annual means & anomalies
    arctic_condition = annual_zonal_mean_datap['latitude'] >= Arctic_min_lat
    arctic_data = annual_zonal_mean_datap.where(arctic_condition, drop=True)
    annual_mean_Arctic = cosine_weighted_spatial_mean(arctic_data)
    Arctic_anom = annual_mean_Arctic - annual_mean_Arctic.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')

    ### Global annual means & anomalies
    annual_mean_global = cosine_weighted_spatial_mean(annual_zonal_mean_datap)
    global_anom = annual_mean_global - annual_mean_global.sel(year=slice(epoch[0], epoch[1])).mean(dim='year')

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
    fig, ax = plt.subplots(figsize=(6.5, 5))

    ### Adjust axes
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both', length=5.5, width=2, which='major')

    ### Ticks and domain
    x_ticks = np.arange(1940, 2031, 10)
    y_ticks = np.arange(-100, 101, 20)
    plt.xticks(x_ticks, x_ticks, color='k')
    plt.yticks(y_ticks, map(str, y_ticks), color='k')
    plt.xlim([1940, 2025]); plt.ylim([-80, 60])

    ### Title
    plt.title(r'\textbf{ARCTIC AMPLIFICATION}', color='k', fontsize=25)

    ### Horizontal lines
    for step, style, lw in [(20, 'solid', .5), (10, 'dotted', .5)]:
        for y in np.arange(-100, 100, step):
            plt.axhline(y, color='gray', linestyle=style, linewidth=1 if y == 0 else lw)

    ### Plot
    datasets = ['Global', rf'Arctic ($\geq${Arctic_min_lat}°N)']
    years = np.arange(year_min, year_max + 1, 1)
    plt.plot(years, global_anom['PRECT'], color='k', linewidth=3, label=rf'\textbf{{{datasets[0]}}}')
    plt.plot(years, Arctic_anom['PRECT'], color='crimson', linewidth=3, label=rf'\textbf{{{datasets[1]}}}')

    ### Labels, annotation, and legend
    plt.xlabel(r'\textbf{Year}', fontsize=13, color='k', labelpad=10)
    plt.ylabel(r'\textbf{Precipitation Anomaly [mm yr\textsuperscript{-1}]}', fontsize=13, color='k', labelpad=10)
    plt.text(2025, -78, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}}}', fontsize=11, rotation='horizontal', ha='right', color='k')
    l = plt.legend(shadow=False, fontsize=11, loc='upper center', bbox_to_anchor=(.5, 1.015), fancybox=True, ncol=2, frameon=False)
    for text in l.get_texts():
        text.set_color('k')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

    ### Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_AA_PRECT_{year_min}–{year_max}_{Arctic_min_lat}N.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')