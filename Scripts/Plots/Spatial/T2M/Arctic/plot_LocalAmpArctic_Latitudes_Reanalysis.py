"""
Plot "Arctic Amplification," as in Sweeney et al. (2023), by latitude and initial year, from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
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
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Arctic/T2M/Local AA/Latitudes/')

print(f'\n---- Arctic Amplification (T2M) ----\n')

### Years
year_min, year_max = 1940, 2025

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Load data
filename = os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_T2M_Monthly.nc')
datat = xr.open_dataset(filename, chunks={'time': 12})

### Process data
T2M = datat['t2m'].groupby('time.year').mean(dim='time').sel(year=slice(year_min, year_max))
T2M_global = T2M.weighted(np.cos(np.deg2rad(T2M.latitude))).mean(dim=('latitude', 'longitude'))

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### Figure
    print(f'{text_months}')

    ### Process data
    if text_months == 'ANNUAL':
        T2M = datat['t2m'].groupby('time.year').mean(dim='time').sel(year=slice(year_min, year_max))
    elif text_months == 'DJF':
        da = datat['t2m'].sel(time=datat.time.dt.month.isin(mons))
        winter_year = xr.where(da.time.dt.month == 12, da.time.dt.year + 1, da.time.dt.year)
        da = da.assign_coords(winter_year=('time', winter_year.data))
        T2M = da.groupby('winter_year').mean(dim='time').rename({'winter_year': 'year'}).sel(year=slice(year_min + 1, year_max))
    else:
        T2M = datat['t2m'].sel(time=datat.time.dt.month.isin(mons)).groupby('time.year').mean(dim='time').sel(year=slice(year_min, year_max))
    T2M_global = T2M.weighted(np.cos(np.deg2rad(T2M.latitude))).mean(dim=('latitude', 'longitude'))

    ### Initial years and Arctic latitudes
    first_year = year_min + 1 if text_months == 'DJF' else year_min
    initial_years = np.arange(first_year, 2001, 5)
    Arctic_min_lats, min_global_rate = [60, 65, 66.56, 70, 75, 80, 85], .05  # °C decade⁻¹ below which the unstable ratio is masked

    ### Global warming rates
    global_series, global_slopes = [], np.empty(len(initial_years))
    global_rates = np.empty(len(initial_years))
    for j, start_year in enumerate(initial_years):
        global_ts = T2M_global.sel(year=slice(start_year, T2M.year.max().item()))
        global_series.append(global_ts)
        global_slopes[j] = np.polyfit(global_ts.year, global_ts, 1)[0]
        global_rates[j] = 10 * global_slopes[j]  # °C decade⁻¹

    ### Create array of amplification values
    AA = np.empty((len(Arctic_min_lats), len(initial_years)))
    for i, lat0 in enumerate(Arctic_min_lats):
        T2M_arctic = T2M.sel(latitude=slice(90, lat0))
        weights_arctic = np.cos(np.deg2rad(T2M_arctic.latitude))
        T2M_arctic = T2M_arctic.weighted(weights_arctic).mean(dim=('latitude', 'longitude'))
        for j in range(len(initial_years)):
            global_ts = global_series[j]
            arctic_ts = T2M_arctic.sel(year=global_ts.year)
            arctic_slope = np.polyfit(global_ts.year, arctic_ts, 1)[0]
            AA[i, j] = arctic_slope / global_slopes[j] if global_rates[j] >= min_global_rate else np.nan

    ### Figure design
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='w')
    plt.rc('axes', edgecolor='k', labelcolor='k')
    plt.rc('xtick', color='k')
    plt.rc('ytick', color='k')

    ### Create figure
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_frame_on(False)

    ### Colors
    if text_months == 'ANNUAL':
        levels, cbar_ticks = np.arange(1, 4.1, .5), np.arange(1, 5, 1)
    elif text_months == 'DJF':
        levels, cbar_ticks = np.arange(1, 6.1, .5), np.arange(1, 7, 1)
    else:
        levels, cbar_ticks = np.arange(0, 2.01, .25), np.arange(0, 3, 1)
    cmap, norm = plt.get_cmap('viridis', len(levels)), plt.matplotlib.colors.BoundaryNorm(levels, len(levels), extend='max')

    ### Plot
    nx, ny = len(initial_years), len(Arctic_min_lats)
    nxs, nys = np.arange(nx + 1), np.arange(ny + 1)
    im = ax.pcolormesh(nxs, nys, AA, cmap=cmap, norm=norm, edgecolors='w', linewidth=1)
    for i in range(ny):
        for j in range(nx):
            if np.isfinite(AA[i, j]):
                ax.text(j + .5, i + .5, f'{AA[i,j]:.2f}', ha='center', va='center', fontsize=6, color='w')

    ### Ticks
    tick_idx = np.arange(0, nx, 2)
    ax.set_xticks(tick_idx + .5)
    ax.set_xticklabels(initial_years[tick_idx])
    ax.set_yticks(np.arange(ny) + .5)
    ax.set_yticklabels([f'{lat}°N' for lat in Arctic_min_lats])
    ax.tick_params(axis='both', which='major', length=0, labelsize=8)

    ### Domain
    ax.set_xlim(0, nx); ax.set_ylim(0, ny)

    ### Title
    ax.set_title(rf'\textbf{{Arctic Amplification}}', fontsize=18, y=1.05)

    ### Labels
    ax.set_xlabel(r'\textbf{Initial Year}', fontsize=9)
    ax.set_ylabel(r'\textbf{Arctic Definition}', fontsize=9)

    ### Annotations
    month_text = '' if text_months == 'ANNUAL' else f'{text_months}'
    fig.text(.08, 1, month_text, fontsize=13, fontweight='bold', color='dimgray', ha='right', va='top')
    fig.text(.85, 1, f'–{int(T2M.year.max())}', fontsize=13, fontweight='bold', color='dimgray', ha='right', va='top')

    ### Annotation for global warming rates
    thresholds = np.arange(.15, .5, .05)
    for threshold in thresholds:
        idx = np.where(global_rates >= threshold)[0]
        if len(idx) == 0:
            continue
        x = idx[0]
        ax.axvline(x=x, color='r', linewidth=.5, ymin=0, ymax=1.04, clip_on=False)
        x_pos, y_pos = x + .9, ny + .08
        threshold_str = f'{threshold:.2f}'.lstrip('0')
        annotation = rf'$\geq${threshold_str}°C decade\textsuperscript{{-1}}'
        ax.text(x_pos, y_pos, annotation, ha='center', va='bottom', fontsize=4, color='r')

    ### Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', aspect=18, pad=.05, ticks=cbar_ticks, extend='max')
    cbar.ax.minorticks_off()
    cbar.outline.set(linewidth=1, edgecolor='k')
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.set_yticklabels([f'{tick:g}' for tick in cbar_ticks], fontsize=8)
    cbar.ax.set_ylabel(r'\textbf{Arctic-Mean Trend / Global-Mean Trend}', rotation=270, labelpad=16, fontsize=9)

    ### Save figure
    fig_path = os.path.join(fig_dir, f'Reanalysis_Arctic_T2M_AA_Latitudes_{text_months}_{year_min}–{year_max}.png')
    plt.savefig(fig_path, dpi=1000, bbox_inches='tight')
    plt.close(fig)

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')