"""
Plot change in 2-m Air Temperature (T2M) over the contiguous United States from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from scipy.stats import linregress, t as student_t
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np
import os
import time
import xarray as xr

mpl.use('Agg')  # non-interactive backend to avoid tkinter issues

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
fig_dir = os.path.join(base_dir, 'Figures/Spatial/CONUS/T2M/Slope/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- CONUS Warming (T2M) ----\n')

### Load data
datat = xr.open_dataset(os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_T2M_Monthly.nc'))['t2m']

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each initial year option
for start_year in range(year_min, 2001, 5):

    ### Loop through each set of months
    for sel_month, mons in months_options.items():

        ### String corrections
        text_months = sel_month.split()[1]

        ### First valid year
        first_year = start_year + 1 if text_months == 'DJF' else start_year

        ### Figure
        print(f'{first_year} | {text_months}')

        ### Process data over CONUS plus margin, before FDR
        T2M = datat.sel(time=datat['time'].dt.month.isin(mons)).where((datat.latitude >= 20) & (datat.latitude <= 55) & (datat.longitude >= 230) & (datat.longitude <= 300), drop=True)
        if text_months == 'DJF':
            month, year = T2M.time.dt.month, T2M.time.dt.year
            winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
            T2M = T2M.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
            T2M = T2M.groupby('winter_year').mean(dim='time').sel(winter_year=slice(first_year, year_max)).rename({'winter_year': 'year'})
        else:
            T2M = T2M.groupby('time.year').mean(dim='time').sel(year=slice(first_year, year_max))

        ### Compute slope and p-value
        T2M = T2M.chunk({'year': -1})  # year rechunking
        time_index = T2M.year.values
        slopes, p_values = xr.apply_ufunc(slope_pvalue, time_index, T2M, input_core_dims=[['year'], ['year']], output_core_dims=[[], []], vectorize=True, dask='parallelized')

        ### Area-weighted FDR correction
        sig_threshold, slopes, p_values = 2 * .05, slopes.compute(), p_values.compute()  # alpha_FDR = 0.10 controls field significance at the 95% confidence level (Wilks 2016)
        weights = np.cos(np.deg2rad(p_values['latitude'].values))
        weights /= np.nanmean(weights)
        p, w = p_values.values.flatten(), np.broadcast_to(weights[:, None], p_values.shape).flatten()
        p, w = p[np.isfinite(p)], w[np.isfinite(p)]
        p_sorted, w_sorted = p[np.argsort(p)], w[np.argsort(p)]
        passed = p_sorted <= (np.cumsum(w_sorted) / np.sum(w_sorted) * sig_threshold)
        cutoff = p_sorted[np.max(np.where(passed))] if np.any(passed) else .0
        pvalues_fdr = xr.DataArray(np.where(np.isfinite(p_values), p_values, np.nan), dims=p_values.dims, coords=p_values.coords)
        sig_mask = pvalues_fdr <= cutoff

        ### Figure design
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
        plt.rc('savefig', facecolor='w')
        plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='dimgray')
        plt.rc('xtick', color='dimgray')
        plt.rc('ytick', color='dimgray')

        ### Create figure
        fig = plt.figure(figsize=(6.5, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=-96))
        ax.set_extent([-124, -68, 24, 50], crs=ccrs.PlateCarree())
        ax.set_xticks([])
        ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontsize=5); ax.xaxis.set_tick_params(labelsize=5); ax.yaxis.set_tick_params(labelsize=5)
        ax.coastlines()
        gridlines = ax.gridlines(draw_labels=True, linewidth=0); gridlines.xlines = gridlines.ylines = False; gridlines.xlabel_style = gridlines.ylabel_style = {'size': 7}
        ax.add_feature(cfeature.LAND, facecolor='none', linewidth=.5, zorder=2); ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='50m', edgecolor='k', facecolor='none'), zorder=3, linewidth=.5)
        ax.add_feature(cfeature.NaturalEarthFeature(category='cultural', name='admin_1_states_provinces_lines', scale='10m', facecolor='none'), zorder=3, edgecolor='dimgray', linewidth=.5)

        ### Plot
        x, y = np.meshgrid(slopes.longitude, slopes.latitude)
        cf = ax.contourf(x, y, slopes, levels=np.arange(-1, 1.1, .1), cmap=plt.get_cmap('RdBu_r', lut=192), vmin=-1, vmax=1, transform=ccrs.PlateCarree(), extend='both')

        ### Colorbar
        cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.12, shrink=.8, ticks=np.arange(-1, 1.01, 1), extend='both')
        cbar.ax.tick_params(labelsize=10, color='dimgray', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
        cbar.ax.set_xticks(np.linspace(-1, 1, 11)); cbar.ax.set_xticklabels(['-1', '-.8', '-.6', '-.4', '-.2', '0', '+.2', '+.4', '+.6', '+.8', '+1'])
        cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.set_xlabel(r'\textbf{Change in 2-m Air Temperature [°C decade\textsuperscript{-1}]}', fontsize=13, color='k', labelpad=7)

        ### Title and annotation
        plt.title(r'\textbf{SURFACE AIR TEMPERATURE CHANGE}', color='k', fontsize=20, y=1.16)
        plt.text(.5, 1.15, rf'\textbf{{{"" if text_months == "ANNUAL" else f"{text_months} "}{first_year}–{year_max}}}', fontsize=12, ha='center', va='top', color='k', transform=ax.transAxes)

        ### Ensure outline is on top
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none', edgecolor='dimgray', linewidth=1.5, zorder=10))

        ### Save figure without stippling
        plt.subplots_adjust(top=.82, bottom=.04, left=.06, right=.94)
        plt.savefig(os.path.join(fig_dir, 'no_stippling', f'Reanalysis_CONUS_T2M_Change_{text_months}_{first_year}–{year_max}_no_stippling.png'), dpi=1000)

        ### Stippling
        plt.rc('hatch', color='silver', linewidth=.5)
        ax.contourf(x, y, sig_mask, levels=[.5, 1.5], hatches=['...', None], colors='none', transform=ccrs.PlateCarree(), zorder=5)
        plt.text(.5, -.13, r'\textbf{\textit{Stippling indicates trends are significant at the 95\% confidence level following area-weighted FDR correction}}',
                 fontsize=5, rotation='horizontal', ha='center', color='silver', alpha=1, transform=ax.transAxes, va='bottom')

        ### Save figure
        plt.savefig(os.path.join(fig_dir, f'Reanalysis_CONUS_T2M_Change_{text_months}_{first_year}–{year_max}.png'), dpi=1000)
        plt.close()

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')