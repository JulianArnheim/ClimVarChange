"""
Plot change in the 200 hPa vertical pressure velocity (OMEGA200) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from scipy.stats import linregress, t as student_t
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
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Global/OMEGA200/Slope/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Change in 200 hPa Vertical Pressure Velocity (OMEGA200) ----\n')

### Load data
datap = xr.open_dataset(os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_OMEGA200_Monthly.nc'))['OMEGA200']

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each initial year option
for start_year in range(year_min, 1981, 10):

    ### Loop through each set of months
    for sel_month, mons in months_options.items():

        ### String corrections
        text_months = sel_month.split()[1]

        ### First valid year
        first_year = start_year + 1 if text_months == 'DJF' else start_year

        ### Figure
        print(f'{first_year} | {text_months}')

        ### Process data
        OMEGA200 = datap.sel(time=datap['time'].dt.month.isin(mons))
        if text_months == 'DJF':
            month, year = OMEGA200.time.dt.month, OMEGA200.time.dt.year
            winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
            OMEGA200 = OMEGA200.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
            OMEGA200 = OMEGA200.groupby('winter_year').mean(dim='time').sel(winter_year=slice(first_year, year_max)).rename({'winter_year': 'year'})
        else:
            OMEGA200 = OMEGA200.groupby('time.year').mean(dim='time').sel(year=slice(first_year, year_max))

        ### Compute slope and p-value
        OMEGA200 = OMEGA200.chunk({'year': -1})  # year rechunking
        time_index = OMEGA200.year.values
        slopes, p_values = xr.apply_ufunc(slope_pvalue, time_index, OMEGA200, input_core_dims=[['year'], ['year']], output_core_dims=[[], []], vectorize=True, dask='parallelized')

        ### Area-weighted FDR correction
        sig_threshold, p_values = 2 * .05, p_values.compute()  # alpha_FDR = 0.10 controls field significance at the 95% confidence level (Wilks 2016)
        weights = np.cos(np.deg2rad(p_values['latitude'].values))
        weights /= np.nanmean(weights)
        p, w = p_values.values.flatten(), np.broadcast_to(weights[:, None], p_values.shape).flatten()
        p, w = p[np.isfinite(p)], w[np.isfinite(p)]
        p_sorted, w_sorted = p[np.argsort(p)], w[np.argsort(p)]
        passed = p_sorted <= (np.cumsum(w_sorted) / np.sum(w_sorted) * sig_threshold)
        cutoff = p_sorted[np.max(np.where(passed))] if np.any(passed) else .0
        valid_pvalues = np.where(np.isfinite(p_values), p_values, np.nan)
        pvalues_fdr = xr.DataArray(valid_pvalues, dims=p_values.dims, coords=p_values.coords)
        insig_mask = pvalues_fdr > cutoff

        ### Bring into memory
        slopes, pvalues_fdr = slopes.compute(), pvalues_fdr.compute()

        ### Vertical pressure velocity change statistics (Pa s⁻¹ decade⁻¹) and area without statistical significance
        weights = np.cos(np.deg2rad(slopes.latitude))
        Mean = slopes.weighted(weights).mean(dim=('latitude', 'longitude')).values
        Min, Max = slopes.min().values, slopes.max().values
        print(f'Global [Min/Mean/Max]: {Min:+.4e} / {Mean:+.4e} / {Max:+.4e} Pa s⁻¹ decade⁻¹')
        frac_insignificant = (insig_mask).weighted(weights).mean(dim=('latitude', 'longitude')).values
        print(f'Area without statistical significance: {frac_insignificant:.2%}')

        ### Figure design
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
        plt.rc('savefig', facecolor='w')
        plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='dimgray')
        plt.rc('xtick', color='dimgray')
        plt.rc('ytick', color='dimgray')

        ### Create figure
        fig = plt.figure(figsize=(6, 4.5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.set_xticks([]); ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontsize=5)
        ax.coastlines()
        ax.add_feature(cfeature.LAND, facecolor='none', linewidth=.5, zorder=2)
        ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
        gridlines = ax.gridlines(draw_labels=True, linewidth=.5, zorder=4); gridlines.xlabel_style = gridlines.ylabel_style = {'size': 7}

        ### Plot
        x, y = np.meshgrid(datap.longitude, datap.latitude)
        cf = ax.contourf(x, y, slopes, levels=np.linspace(-3e-3, 3e-3, 13), cmap=plt.get_cmap('BrBG_r', lut=192), vmin=-3e-3, vmax=3e-3, transform=ccrs.PlateCarree(), extend='both')
        ax.contourf(x, y, insig_mask, levels=[.5, 1.5], colors='w', transform=ccrs.PlateCarree(), zorder=1)  # uncertainty mask

        ### Colorbar
        cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.1, shrink=.8, extend='both')
        cbar.ax.tick_params(labelsize=10, color='dimgray', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
        cbar.ax.set_xticks(np.linspace(-3e-3, 3e-3, 7)); cbar.ax.set_xticklabels(['-3', '-2', '-1', '0', '+1', '+2', '+3'])
        cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.set_xlabel(r'\textbf{Change in 200 hPa Vertical Pressure Velocity [10\textsuperscript{-3} Pa s\textsuperscript{-1} decade\textsuperscript{-1}]}', fontsize=13, color='k', labelpad=8)

        ### Title and annotations
        plt.title(r'\textbf{VERTICAL PRESSURE VELOCITY CHANGE}', color='k', fontsize=20, y=1.12)
        season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
        plt.text(.88, 1.08, rf'\textbf{{{season_prefix}{first_year}–{year_max}}}', fontsize=10, rotation='horizontal', ha='left', va='top', color='k', transform=ax.transAxes)
        plt.text(.5, -.13, r'\textbf{\textit{Shading indicates trends are significant at the 95\% confidence level following area-weighted FDR correction}}',
                 fontsize=5, rotation='horizontal', ha='center', color='dimgray', alpha=1, transform=ax.transAxes, va='bottom')

        ### Save figure
        plt.subplots_adjust(top=.98, bottom=.03, left=.08, right=.92)
        plt.savefig(os.path.join(fig_dir, f'Reanalysis_Global_OMEGA200_Change_{text_months}_{first_year}–{year_max}.png'), dpi=1000)
        plt.close()

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')