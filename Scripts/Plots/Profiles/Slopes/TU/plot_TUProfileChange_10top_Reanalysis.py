"""
Plot change in Temperature (T) and Zonal Wind (U) profile from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from scipy.stats import linregress, t as student_t
import dask.config as dask_config
import matplotlib.colors as mcolors, matplotlib.pyplot as plt
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
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Profile/Slopes/Global/10 top/TU/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Global Warming & Zonal Wind Response ----\n')

### Load T data
datat = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_Global_{start_year}-{end_year}_TProfile_Monthly.nc')
    if os.path.exists(filename):
        ds = xr.open_dataset(filename, chunks='auto').astype(np.float32)
        datat.append(ds)
        print(f'Loaded T: {start_year}–{end_year}')
if datat:
    datat = xr.concat(datat, dim='time').sortby('time')
else:
    raise FileNotFoundError('No ERA5_Global_*_TProfile_Monthly.nc files found.')
print()

### Load U data
datau = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_Global_{start_year}-{end_year}_UProfile_Monthly.nc')
    if os.path.exists(filename):
        ds = xr.open_dataset(filename, chunks='auto').astype(np.float32)
        datau.append(ds)
        print(f'Loaded U: {start_year}–{end_year}')
if datau:
    datau = xr.concat(datau, dim='time').sortby('time')
else:
    raise FileNotFoundError('No ERA5_Global_*_UProfile_Monthly.nc files found.')
print()

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Sector settings
longitudes_options = {'1. NH (180°W–180°E)': lambda lon: (lon >= 0) & (lon < 360),
                      '2. AM (140°W–50°W)': lambda lon: (lon >= 360 - 140) & (lon < 360 - 50),  # -140 to -50
                      '3. AT (50°W–40°E)': lambda lon: ((lon >= 360 - 50) & (lon < 360)) | ((lon >= 0) & (lon <= 40)),  # -50 to 40
                      '4. AS (40°E–130°E)': lambda lon: (lon >= 40) & (lon <= 130),  # 40 to 130
                      '5. PA (130°E–140°W)': lambda lon: (lon >= 130) & (lon < 360 - 140)}  # 130 to -140

### Loop through each initial year option
for start_year in range(year_min, 1990, 10):

    ### Loop through each set of months
    for sel_month, mons in months_options.items():

        ### Loop through each sector
        for sel_longitude, lon_func in longitudes_options.items():

            ### String corrections
            text_months, text_longitudes = sel_month.split()[1], sel_longitude.split()[1]

            ### First valid year
            first_year = start_year + 1 if text_months == 'DJF' else start_year

            ### Figure
            print(f'{first_year} | {text_months} | {text_longitudes}')

            ### ----- TEMPERATURE (T) -----
            annual_zonal_mean_datat_filename = os.path.join(data_dir, f'ERA5_Global_{first_year}-{year_max}_TProfile_{text_months}_{text_longitudes}_AnnualZonalMeans.nc')
            if os.path.exists(annual_zonal_mean_datat_filename):
                annual_zonal_mean_datat = xr.open_dataset(annual_zonal_mean_datat_filename)
                if 'winter_year' in annual_zonal_mean_datat:
                    annual_zonal_mean_datat = annual_zonal_mean_datat.rename({'winter_year': 'year'})
                print(f'Using Previously Computed ERA5 T Annual Zonal Means')
            else:
                print(f'File being made: {annual_zonal_mean_datat_filename}')
                datat_sel = datat.sel(time=(datat['time'].dt.month.isin(mons) & (datat['time'].dt.year >= year_min))).where(lon_func(datat.longitude), drop=True)
                if text_months == 'DJF':
                    month, year = datat_sel['time'].dt.month, datat_sel['time'].dt.year
                    winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
                    datat_sel = datat_sel.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
                    annual_zonal_mean_datat = datat_sel.mean(dim='longitude').groupby('winter_year').mean(dim='time').sel(winter_year=slice(start_year + 1, year_max)).rename({'winter_year': 'year'})
                else:
                    annual_zonal_mean_datat = datat_sel.mean(dim='longitude').groupby('time.year').mean(dim='time').sel(year=slice(start_year, year_max))
                annual_zonal_mean_datat.to_netcdf(annual_zonal_mean_datat_filename)
                print(f'Saved {annual_zonal_mean_datat_filename}, the ERA5 T Annual Zonal Means')

            ### Calculate and save annual zonal mean slopes
            T_slopes_filename = os.path.join(data_dir, f'ERA5_Global_{first_year}-{year_max}_TProfile_Slopes_{text_months}_{text_longitudes}.nc')
            T_pvalues_filename = os.path.join(data_dir, f'ERA5_Global_{first_year}-{year_max}_TProfile_PValues_{text_months}_{text_longitudes}.nc')
            if os.path.exists(T_slopes_filename) and os.path.exists(T_pvalues_filename):
                T_slopes, T_pvalues = xr.open_dataset(T_slopes_filename), xr.open_dataset(T_pvalues_filename)
                print(f'Using Previously Computed ERA5 T Annual Zonal Mean Slopes and P-Values')
            else:
                print(f'Files being made: {T_slopes_filename} & {T_pvalues_filename}')
                annual_zonal_mean_datat = annual_zonal_mean_datat.sel(year=slice(first_year, None))
                years = annual_zonal_mean_datat['year'].values
                with dask_config.set(**{'array.slicing.split_large_chunks': True}):
                    T_slopes, T_pvalues = xr.apply_ufunc(slope_pvalue, years, annual_zonal_mean_datat, input_core_dims=[['year'], ['year']], output_core_dims=[[], []],
                                                         output_dtypes=[np.float32, np.float32], vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk': True})
                T_slopes.to_netcdf(T_slopes_filename)
                T_pvalues.to_netcdf(T_pvalues_filename)
                print(f'Saved {T_slopes_filename} and {T_pvalues_filename}, the ERA5 T Annual Zonal Mean Slopes and P-Values')

            ### Area- and pressure-weighted FDR correction over the plotted domain
            sig_threshold = 2 * .05  # alpha_FDR = 0.10 controls field significance at the 95% confidence level (Wilks 2016)
            weights = np.cos(np.deg2rad(T_pvalues['latitude'].values))[None, :] * np.abs(np.gradient(T_pvalues['level'].values))[:, None]  # area and pressure-thickness weights
            weights /= np.nanmean(weights)
            p = T_pvalues['t'].where(T_pvalues['level'] >= 10).values.flatten()  # plotted domain only
            w = np.broadcast_to(weights, T_pvalues['t'].shape).flatten()
            p, w = p[np.isfinite(p)], w[np.isfinite(p)]
            p_sorted, w_sorted = p[np.argsort(p)], w[np.argsort(p)]
            passed = p_sorted <= (np.cumsum(w_sorted) / np.sum(w_sorted) * sig_threshold)
            valid_pvalues = np.where(np.isfinite(T_pvalues['t']), T_pvalues['t'], np.nan)
            pvalues_fdr = xr.DataArray(valid_pvalues, dims=['level', 'latitude'], coords={'level': T_pvalues['level'], 'latitude': T_pvalues['latitude']}, name='t')
            insig_mask = pvalues_fdr > (p_sorted[np.max(np.where(passed))] if np.any(passed) else .0)

            ### ----- ZONAL WIND (U) -----
            annual_zonal_mean_datau_filename = os.path.join(data_dir, f'ERA5_Global_{first_year}-{year_max}_UProfile_{text_months}_{text_longitudes}_AnnualZonalMeans.nc')
            if os.path.exists(annual_zonal_mean_datau_filename):
                annual_zonal_mean_datau = xr.open_dataset(annual_zonal_mean_datau_filename)
                if 'winter_year' in annual_zonal_mean_datau:
                    annual_zonal_mean_datau = annual_zonal_mean_datau.rename({'winter_year': 'year'})
                print(f'Using Previously Computed ERA5 U Annual Zonal Means')
            else:
                print(f'File being made: {annual_zonal_mean_datau_filename}')
                datau_sel = datau.sel(time=(datau['time'].dt.month.isin(mons) & (datau['time'].dt.year >= year_min))).where(lon_func(datau.longitude), drop=True)
                if text_months == 'DJF':
                    month, year = datau_sel['time'].dt.month, datau_sel['time'].dt.year
                    winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
                    datau_sel = datau_sel.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
                    annual_zonal_mean_datau = datau_sel.mean(dim='longitude').groupby('winter_year').mean(dim='time').sel(winter_year=slice(start_year + 1, year_max)).rename({'winter_year': 'year'})
                else:
                    annual_zonal_mean_datau = datau_sel.mean(dim='longitude').groupby('time.year').mean(dim='time').sel(year=slice(start_year, year_max))
                annual_zonal_mean_datau.to_netcdf(annual_zonal_mean_datau_filename)
                print(f'Saved {annual_zonal_mean_datau_filename}, the ERA5 U Annual Zonal Means')

            ### Calculate and save annual zonal mean slopes
            U_slopes_filename = os.path.join(data_dir, f'ERA5_Global_{first_year}-{year_max}_UProfile_Slopes_{text_months}_{text_longitudes}.nc')
            if os.path.exists(U_slopes_filename):
                U_slopes = xr.open_dataset(U_slopes_filename)
                print(f'Using Previously Computed ERA5 U Annual Zonal Mean Slopes')
            else:
                print(f'File being made: {U_slopes_filename}')
                annual_zonal_mean_datau = annual_zonal_mean_datau.sel(year=slice(first_year, None))
                years = annual_zonal_mean_datau['year'].values
                with dask_config.set(**{'array.slicing.split_large_chunks': True}):
                    U_slopes, _ = xr.apply_ufunc(slope_pvalue, years, annual_zonal_mean_datau, input_core_dims=[['year'], ['year']], output_core_dims=[[], []],
                                                 output_dtypes=[np.float32, np.float32], vectorize=True, dask='parallelized', dask_gufunc_kwargs={'allow_rechunk': True})
                U_slopes.to_netcdf(U_slopes_filename)
                print(f'Saved {U_slopes_filename}, the ERA5 U Annual Zonal Mean Slopes')

            ### Figure design
            plt.rc('text', usetex=True)
            plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
            plt.rc('savefig', facecolor='w')
            plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='k')
            plt.rc('xtick', color='dimgray', labelcolor='k')
            plt.rc('ytick', color='dimgray', labelcolor='k')

            ### Create figure
            fig, ax = plt.subplots(figsize=(9, 5))

            ### Spines
            for side in ['top', 'right', 'bottom', 'left']:
                ax.spines[side].set_color('dimgray')
                ax.spines[side].set_linewidth(2)
                ax.spines[side].set_zorder(10)

            ### Ticks
            xtick = np.arange(-90, 91, 15)
            zscale = np.array([1000, 700, 500, 300, 200, 100, 50, 30, 10])
            ax.tick_params(axis='both', direction='out', which='major', pad=3, width=2, color='dimgray')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            plt.yscale('log')
            plt.minorticks_off()
            plt.gca().invert_yaxis()
            ax.invert_yaxis()
            ax.set_xticks(xtick)
            ax.set_yticks(zscale)

            ### Tick labels
            ax.set_xticklabels([f'{tick}°' for tick in xtick], fontname='Avant Garde', fontsize=11)
            ax.set_yticklabels([str(p) for p in zscale], fontname='Avant Garde', fontsize=11)

            ### Domain
            ax.set(xlim=[-90, 90], ylim=[1000, 10])

            ### Uncertainty mask
            X, Y = np.meshgrid(pvalues_fdr['latitude'], pvalues_fdr['level'])
            ax.contourf(X, Y, np.where(insig_mask.values, 1, np.nan), levels=[.5, 1], colors='w', zorder=3)

            ### Custom color (T)
            cmap = plt.get_cmap('RdBu_r')
            levels = np.arange(-.5, .55, .05)
            value_positions = np.concatenate([[-1], levels, [1]])
            norm = mcolors.BoundaryNorm(value_positions, cmap.N)

            ### Plot
            im = ax.contourf(T_slopes['latitude'], T_slopes['level'], T_slopes.t, norm=norm, cmap=cmap, levels=levels, extend='both')

            ### Colorbar (T)
            cbar = plt.colorbar(im, ax=ax, orientation='horizontal', aspect=40, pad=.16, shrink=.8)
            cbar.ax.tick_params(labelsize=12, labelright=True, labelleft=False, right=True, left=False, color='dimgray')
            cbar.ax.set_xticks(np.linspace(-.5, .5, 11))
            cbar.ax.set_xticklabels(['-.5', '-.4', '-.3', '-.2', '-.1', '0', '+.1', '+.2', '+.3', '+.4', '+.5'], color='k')
            cbar.ax.minorticks_off()

            ### Contours (U)
            contour_levels = np.concatenate([np.arange(-10, 0, .5), np.arange(.5, 10, .5)])
            fmt = lambda x: f'{x:+.1f}'.replace('+0.', '+.').replace('-0.', '-.').rstrip('0').rstrip('.')
            CS = ax.contour(U_slopes['latitude'], U_slopes['level'], U_slopes['u'], levels=[l for l in contour_levels if l != 0], colors='k', linewidths=1.25, zorder=5)
            clabels = ax.clabel(CS, CS.levels, inline=True, fontsize=8, fmt=fmt, colors='k', manual=False, rightside_up=True, use_clabeltext=True)

            ### Both parameters mentioned in colorbar label
            xlabel = r'\textbf{Change in Temperature [°C decade\textsuperscript{-1}] and Zonal Wind [m s\textsuperscript{-1} decade\textsuperscript{-1}]}'
            cbar.ax.set_xlabel(xlabel, fontsize=14, color='k', labelpad=4)

            ### Title
            plt.title(rf'\textbf{{{sel_longitude[2:]} TEMPERATURE AND ZONAL WIND TRENDS $(dT/dt, dU/dt)$}}', color='k', fontsize=16, y=1.06)

            ### Labels
            plt.xlabel(r'\textbf{Latitude [°N]}', color='k', fontsize=15)
            plt.ylabel(r'\textbf{Pressure [hPa]}', color='k', fontsize=15)

            ### Annotation
            season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
            plt.text(1, 1.05, rf'\textbf{{{season_prefix}{first_year}–{year_max}}}', fontsize=10, rotation='horizontal', ha='right', va='top', color='k', transform=ax.transAxes)
            plt.text(1, -.12, r'\textbf{\textit{Shading indicates trends are significant at the 95\% confidence level following area- and pressure-weighted FDR correction}}',
                     fontsize=3, rotation='horizontal', ha='right', color='dimgray', alpha=1, transform=ax.transAxes, va='bottom')

            ### Save figure
            plt.subplots_adjust(bottom=.05)
            plt.savefig(os.path.join(fig_dir, text_months, f'Reanalysis_{first_year}–{year_max}_TUProfileSlope_10top_{text_months}_{text_longitudes}.png'), dpi=1000)
            plt.close()

            ### Progress
            print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')