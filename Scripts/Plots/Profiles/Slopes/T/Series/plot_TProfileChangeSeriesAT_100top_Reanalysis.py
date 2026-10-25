"""
Plot change in Temperature (T) profile for reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from scipy.stats import linregress, t as student_t
import dask.config as dask_config
import matplotlib as mpl, matplotlib.colors as mcolors, matplotlib.pyplot as plt
import numpy as np
import os
import time
import xarray as xr

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

def slope_pvalue(x, y):
    fit = linregress(x, y)
    resid = y - fit.intercept - fit.slope * x
    r1 = max(np.nan_to_num(np.corrcoef(resid[:-1], resid[1:])[0, 1]), 0) if np.std(resid) > 0 else 0  # lag-1 autocorrelation of residuals
    n_eff = max(len(y) * (1 - r1) / (1 + r1), 3)  # effective sample size (Santer et al. 2000)
    return fit.slope * 10, (2 * student_t.sf(abs(fit.slope) / fit.stderr * np.sqrt((n_eff - 2) / (len(y) - 2)), n_eff - 2) if fit.stderr > 0 else fit.pvalue)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Profile/Slopes/Series/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Global Warming ----\n')

### Load data
datat = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_Global_{start_year}-{end_year}_TProfile_Monthly.nc')
    if os.path.exists(filename):
        ds = xr.open_dataset(filename, chunks={'time': 12, 'latitude': 16, 'level': 10}).astype(np.float32)
        datat.append(ds)
        print(f'Loaded: {start_year}–{end_year}')
if datat:
    datat = xr.concat(datat, dim='time').sortby('time')
else:
    raise FileNotFoundError('No ERA5_Global_*_TProfile_Monthly.nc files found.')
print()

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='k')
plt.rc('xtick', color='dimgray', labelcolor='k')
plt.rc('ytick', color='dimgray', labelcolor='k')

### Stippling design
stip_color = '#A9A9A9'  # dark silver
mpl.rcParams['hatch.color'] = stip_color
mpl.rcParams['hatch.linewidth'] = .5

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### First valid year
    first_year = year_min + 1 if text_months == 'DJF' else year_min

    ### Loop through gap size
    gap_sizes = [10, 20]
    for gap in gap_sizes:

        ### Year periods
        yr_periods = [(start_year, year_max) for start_year in range(first_year, first_year + 4 * gap, gap)]

        ### Loop through stippling options
        for stippling in [True, False]:

            ### Figure
            print(f'{text_months} | {gap} year gaps | {"stippling" if stippling else "no stippling"}')

            ### Create figure
            fig, axs = plt.subplots(1, 4, figsize=(6, 1.5), sharex=True, sharey=True)

            ### Loop through each initial year option
            for i, (yr_start, yr_end) in enumerate(yr_periods):

                ### Month and Atlantic sector (50°W–40°E) specification
                datat_season = datat.sel(time=datat['time'].dt.month.isin(mons))
                lon_range = ((datat.longitude >= 310) & (datat.longitude < 360)) | ((datat.longitude >= 0) & (datat.longitude <= 40))

                ### Process seasonal mean
                if text_months == 'DJF':
                    full_year_min = year_min + 1
                    winter_year = xr.where(datat_season['time'].dt.month == 12, datat_season['time'].dt.year + 1, datat_season['time'].dt.year)
                    annual_zonal_mean_datat = datat_season.sel(longitude=lon_range).mean(dim='longitude').assign_coords(winter_year=('time', winter_year.data))
                    annual_zonal_mean_datat = annual_zonal_mean_datat.groupby('winter_year').mean(dim='time').sel(winter_year=slice(full_year_min, year_max)).rename({'winter_year': 'year'})
                else:
                    full_year_min = year_min
                    annual_zonal_mean_datat = datat_season.groupby('time.year').mean(dim='time').sel(longitude=lon_range).mean(dim='longitude')

                ### Calculate and save annual zonal mean slopes
                T_slopes_filename = os.path.join(data_dir, f'ERA5_Global_{yr_start}-{yr_end}_TProfile_Slopes_{text_months}_AT.nc')
                T_pvalues_filename = os.path.join(data_dir, f'ERA5_Global_{yr_start}-{yr_end}_TProfile_PValues_{text_months}_AT.nc')
                if os.path.exists(T_slopes_filename) and os.path.exists(T_pvalues_filename):
                    T_slopes, T_pvalues = xr.open_dataset(T_slopes_filename), xr.open_dataset(T_pvalues_filename)
                    print(f'Using Previously Computed ERA5 AT Zonal Mean Slopes and P-Values')

                else:
                    print(f'Files being made: {T_slopes_filename} & {T_pvalues_filename}')
                    annual_zonal_mean_datat_subset = annual_zonal_mean_datat.sel(year=slice(yr_start, yr_end))
                    years = annual_zonal_mean_datat_subset['year'].values
                    with dask_config.set(**{'array.slicing.split_large_chunks': True}):
                        T_slopes, T_pvalues = xr.apply_ufunc(slope_pvalue, years, annual_zonal_mean_datat_subset, input_core_dims=[['year'], ['year']], output_core_dims=[[], []],
                                                             output_dtypes=[np.float32, np.float32], vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk': True})
                    T_slopes.to_netcdf(T_slopes_filename)
                    T_pvalues.to_netcdf(T_pvalues_filename)
                    print(f'Saved {T_slopes_filename} and {T_pvalues_filename}, the ERA5 AT Zonal Mean Slopes and P-Values')

                ### Area- and pressure-weighted FDR correction over the plotted domain
                sig_threshold = 2 * .05  # alpha_FDR = 0.10 controls field significance at the 95% confidence level (Wilks 2016)
                weights = np.cos(np.deg2rad(T_pvalues['latitude'].values))[None, :] * np.abs(np.gradient(T_pvalues['level'].values))[:, None]  # area and pressure-thickness weights
                weights /= np.nanmean(weights)
                weights_2d = np.broadcast_to(weights, T_pvalues['t'].shape)
                p = T_pvalues['t'].where((T_pvalues['level'] >= 100) & (T_pvalues['latitude'] >= 0)).values.flatten()  # plotted domain only
                w = weights_2d.flatten()
                p, w = p[np.isfinite(p)], w[np.isfinite(p)]
                order = np.argsort(p)
                p_sorted, w_sorted = p[order], w[order]
                threshold = np.cumsum(w_sorted) / np.sum(w_sorted) * sig_threshold
                passed = p_sorted <= threshold
                cutoff = p_sorted[np.max(np.where(passed))] if np.any(passed) else .0
                valid_pvalues = np.where(np.isfinite(T_pvalues['t']), T_pvalues['t'], np.nan)
                pvalues_fdr = xr.DataArray(valid_pvalues, dims=['level', 'latitude'], coords={'level': T_pvalues['level'], 'latitude': T_pvalues['latitude']}, name='t')
                sig_mask = pvalues_fdr <= cutoff

                ### Spines
                ax = axs[i]
                for side in ['top', 'right', 'bottom', 'left']:
                    ax.spines[side].set_color('dimgray')
                    ax.spines[side].set_linewidth(1)
                    ax.spines[side].set_zorder(10)

                ### Ticks and domain
                ax.tick_params(axis='both', direction='out', which='major', pad=1, width=1, length=1.5, color='dimgray')
                ax.xaxis.set_ticks_position('bottom'); ax.yaxis.set_ticks_position('left')
                zscale, xtick_labels = np.array([1000, 700, 500, 300, 200, 100]), np.arange(0, 91, 15)
                plt.xticks(xtick_labels, map(str, xtick_labels), fontsize=4)
                plt.yscale('log')
                plt.minorticks_off()
                plt.gca().invert_yaxis()
                ax.invert_yaxis()
                ax.set_xticks(xtick_labels); ax.set_yticks(zscale)
                ax.set_xticklabels([f'{tick}°' for tick in xtick_labels], fontname='Avant Garde', fontsize=4)
                ax.set_yticklabels([str(p) for p in zscale], fontname='Avant Garde', fontsize=4)
                ax.set(xlim=[0, 90], ylim=[1000, 100])
                if i > 0:
                    ax.tick_params(axis='y', which='both', left=False, labelleft=False)

                ### Colorbar
                cmap = plt.get_cmap('RdBu_r')
                levels = np.arange(-1, 1.1, .1)
                value_positions = np.concatenate([[-1], levels, [1]])

                ### Plot
                norm = mcolors.BoundaryNorm(value_positions, cmap.N)
                im = ax.contourf(T_slopes['latitude'], T_slopes['level'], T_slopes.t, levels=levels, cmap=cmap, norm=norm, extend='both')

                ### Stippling
                if stippling:
                    X, Y = np.meshgrid(pvalues_fdr['latitude'], pvalues_fdr['level'])
                    ax.contourf(X, Y, np.where(sig_mask.values, 1, np.nan), levels=[.5, 1.5], hatches=['.....'], colors='none', zorder=3)

                ### Title
                ax.set_title(rf'\textbf{{{yr_start}–{yr_end}}}', fontsize=7, y=.93)

                ### Labels
                ax.set_xlabel(r'\textbf{Latitude [°N]}', fontsize=5, color='k', labelpad=2)
            axs[0].set_ylabel(r'\textbf{Pressure [hPa]}', fontsize=5, color='k', labelpad=.5)

            ### Colorbar
            cbar = fig.colorbar(im, ax=axs, orientation='horizontal', aspect=50, pad=5, shrink=.75)
            cbar.ax.set_position([cbar.ax.get_position().x0, cbar.ax.get_position().y0 - .02, cbar.ax.get_position().width, cbar.ax.get_position().height])
            cbar.set_ticks(np.linspace(-1, 1, 11))
            cbar.set_ticklabels(['-1', '-.8', '-.6', '-.4', '-.2', '0', '+.2', '+.4', '+.6', '+.8', '+1'])
            cbar.ax.tick_params(axis='x', direction='out', length=1.5, width=.75, pad=1.5, color='dimgray', labelcolor='k', labelsize=5)
            cbar.ax.minorticks_off()
            cbar.outline.set_linewidth(.5)
            cbar.ax.set_xlabel(r'\textbf{Change in Temperature [°C decade\textsuperscript{-1}]}', fontsize=6, color='k', labelpad=1)

            ### Title
            season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
            plt.suptitle(f'{season_prefix}Atlantic Sector (50°W–40°E) Temperature Trend ($dT/dt$)', fontsize=9, color='k', y=.99)

            ### Annotation
            if stippling:
                plt.text(1, -.87, r'\textbf{\textit{Stippling indicates trends are significant at the 95\% confidence level following area- and pressure-weighted FDR correction}}',
                         fontsize=2, rotation='horizontal', ha='right', color=stip_color, alpha=1, transform=ax.transAxes, va='bottom')

            ### Save figure
            stippling_suffix = '' if stippling else '_no_stippling'
            save_fig_dir = fig_dir if stippling else os.path.join(fig_dir, 'no_stippling')
            plt.subplots_adjust(left=.045, right=.99, bottom=.37, top=.78, wspace=.1)
            plt.savefig(os.path.join(save_fig_dir, f'Reanalysis_{full_year_min}–{year_max}_{gap}yr_TProfileSlope_100top_{text_months}_AT_4{stippling_suffix}.png'), dpi=1000)
            plt.close()

            ### Progress
            print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')