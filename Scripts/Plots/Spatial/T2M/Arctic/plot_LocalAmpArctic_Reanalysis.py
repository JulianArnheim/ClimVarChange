"""
Plot "Arctic Amplification," as in Sweeney et al. (2023), by initial and final year, from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.pyplot as plt
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
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Arctic/T2M/Local AA/Periods/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Arctic Amplification (T2M) ----\n')

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

    ### Create array of amplification values
    first_year = year_min + 1 if text_months == 'DJF' else year_min
    initial_years = np.arange(first_year, 2002, 5)
    final_years = np.arange(1990, year_max + 1, 5)
    T2M_arctic = T2M.sel(latitude=slice(90, 70)).weighted(np.cos(np.deg2rad(T2M.latitude))).mean(dim=('latitude', 'longitude'))
    global_slopes = np.full((len(initial_years), len(final_years)), np.nan)
    AA = np.full((len(initial_years), len(final_years)), np.nan)
    required_period, min_global_rate = 30, .05  # years, and °C decade⁻¹ below which the unstable ratio is masked
    for i, start_year in enumerate(initial_years):
        for j, end_year in enumerate(final_years):
            if (end_year - start_year + 1) < required_period:
                continue
            arctic_ts = T2M_arctic.sel(year=slice(start_year, end_year))
            global_ts = T2M_global.sel(year=slice(start_year, end_year))
            if global_ts.sizes['year'] < required_period:
                continue
            arctic_slope = np.polyfit(global_ts.year, arctic_ts, 1)[0]
            global_slope = np.polyfit(global_ts.year, global_ts, 1)[0]
            global_slopes[i, j] = global_slope * 10  # °C decade⁻¹
            AA[i, j] = arctic_slope / global_slope if global_slopes[i, j] >= min_global_rate else np.nan

    ### Figure design
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='w')
    plt.rc('axes', edgecolor='k', labelcolor='k')
    plt.rc('xtick', color='k')
    plt.rc('ytick', color='k')

    ### Create figure
    fig, ax = plt.subplots(figsize=(4.2, 3))
    ax.set_frame_on(False)

    ### Colors
    if text_months == 'ANNUAL':
        levels, cbar_ticks = np.arange(1, 5.1, .25), np.arange(1, 6, 1)
    elif text_months == 'DJF':
        levels, cbar_ticks = np.arange(1, 8.1, .5), np.arange(1, 9, 1)
    else:
        levels, cbar_ticks = np.arange(0, 2.1, .25), np.arange(0, 3, 1)
    cmap = plt.get_cmap('viridis', len(levels))

    ### Plot
    nx, ny = len(final_years), len(initial_years)
    nxs, nys = np.arange(nx + 1), np.arange(ny + 1)
    im = ax.pcolormesh(nxs, nys, AA, cmap=cmap, norm=plt.matplotlib.colors.BoundaryNorm(levels, len(levels), extend='max'), edgecolors='w', linewidth=1)
    for i in range(ny):
        for j in range(nx):
            if np.isfinite(AA[i, j]):
                ax.text(j + .5, i + .5, f'{AA[i,j]:.2f}', ha='center', va='center', fontsize=5, color='w')

    ### Global warming rate cell outlines
    labels, styles_widths = ['$>$0', r'$\geq$.1', r'$\geq$.2'], [((0, (1, 4)), .5), ('--', .5), ('-', .75)]
    for i in range(ny):
        for j in range(nx):
            rate = global_slopes[i, j]
            if np.isfinite(rate) and rate >= 0:
                idx = min(np.digitize(rate, [.1, .2]), 2)
                linestyle, linewidth = styles_widths[idx]
                ax.add_patch(plt.Rectangle((j + .03, i + .04), .94, .92, fill=False, edgecolor='r', linestyle=linestyle, linewidth=linewidth))

    ### Ticks and domain
    tick_idx_x = np.arange(0, nx, 2)
    tick_idx_y = np.arange(0, ny, 2)
    ax.set_xticks(tick_idx_x + .5)
    ax.set_xticklabels(final_years[tick_idx_x])
    ax.set_yticks(tick_idx_y + .5)
    ax.set_yticklabels(initial_years[tick_idx_y])
    if initial_years[-1] not in initial_years[tick_idx_y]:
        ax.set_yticks(np.append(ax.get_yticks(), ny - .5))
        ax.set_yticklabels([*initial_years[tick_idx_y], initial_years[-1]])
    ax.tick_params(axis='both', which='major', length=0, labelsize=8)
    ax.set_xlim(0, nx); ax.set_ylim(0, ny)

    ### Title
    ax.set_title(rf'\textbf{{Arctic Amplification}}', fontsize=15, y=1.08)

    ### Labels and annotations
    ax.set_xlabel(r'\textbf{Final Year}', fontsize=9)
    ax.set_ylabel(r'\textbf{Initial Year}', fontsize=9)
    season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
    fig.text(.06, 1.05, season_prefix.rstrip(), fontsize=10, fontweight='bold', color='dimgray', ha='right', va='top')
    fig.text(.9, .01, rf'{{Arctic: $\geq$70°N}}', fontsize=5, fontweight='bold', color='dimgray', ha='right', va='top')

    ### Legend
    legend = [Line2D([0], [0], color='r', linestyle=s, linewidth=w, label=rf'{l}°C decade\textsuperscript{{-1}}') for l, (s, w) in zip(labels, styles_widths)]
    ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, 1.08), ncols=3, frameon=False, edgecolor='r', fontsize=4, handlelength=2, borderaxespad=.2, borderpad=.5)

    ### Colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', aspect=18, pad=.05, ticks=cbar_ticks, extend='max')
    cbar.ax.minorticks_off()
    cbar.outline.set(linewidth=1, edgecolor='k')
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.set_yticklabels([f'{tick:g}' for tick in cbar_ticks], fontsize=8)
    cbar.ax.set_ylabel(r'\textbf{Arctic-Mean Trend / Global-Mean Trend}', rotation=270, labelpad=16, fontsize=9)

    ### Save figure
    fig_path = os.path.join(fig_dir, f'Reanalysis_Arctic_T2M_AA_Periods_{text_months}_{year_min}–{year_max}.png')
    plt.savefig(fig_path, dpi=1000, bbox_inches='tight')
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')