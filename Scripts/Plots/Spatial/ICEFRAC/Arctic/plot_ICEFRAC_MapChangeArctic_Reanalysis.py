"""
Plot change in Sea Ice Concentration (ICEFRAC) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from cartopy.util import add_cyclic_point
from matplotlib.patches import Rectangle
from scipy.stats import linregress, t as student_t
import cartopy.crs as ccrs, cartopy.feature as cfeature
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
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Arctic/ICEFRAC/Slope/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Global Warming (ICEFRAC) ----\n')

### Load data
datac = xr.open_dataset(os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_ICEFRAC_Monthly.nc'))['siconc']

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
        ICEFRAC = datac.sel(time=datac['time'].dt.month.isin(mons))
        ICEFRAC = ICEFRAC.where(ICEFRAC.latitude >= 30, drop=True) * 100
        if text_months == 'DJF':
            month, year = ICEFRAC.time.dt.month, ICEFRAC.time.dt.year
            winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
            ICEFRAC = ICEFRAC.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
            ICEFRAC = ICEFRAC.groupby('winter_year').mean(dim='time').sel(winter_year=slice(first_year, year_max)).rename({'winter_year': 'year'})
        else:
            ICEFRAC = ICEFRAC.groupby('time.year').mean(dim='time').sel(year=slice(first_year, year_max))

        ### Compute slope and p-value, print sea ice change statistics [% decade⁻¹]
        ICEFRAC = ICEFRAC.chunk({'year': -1})  # year rechunking
        time_index = ICEFRAC.year.values
        slopes, p_values = xr.apply_ufunc(slope_pvalue, time_index, ICEFRAC, input_core_dims=[['year'], ['year']], output_core_dims=[[], []], vectorize=True, dask='parallelized')

        ### Bring into memory
        slopes = slopes.compute()
        pvalues = p_values.compute()

        ### Sea ice change statistics [% decade⁻¹]
        Min, Max = slopes.min().values, slopes.max().values
        print(f'Arctic [Min/Max]: {Min:+.4f} / {Max:+.4f} % decade⁻¹')

        ### Area-weighted FDR correction
        sig_threshold = 2 * .05  # alpha_FDR = 0.10 controls field significance at the 95% confidence level (Wilks 2016)
        p_values = p_values.compute()
        weights = np.cos(np.deg2rad(p_values['latitude'].values))
        weights /= np.nanmean(weights)
        p = p_values.values.flatten()
        w = np.broadcast_to(weights[:, None], p_values.shape).flatten()
        p, w = p[np.isfinite(p)], w[np.isfinite(p)]
        p_sorted, w_sorted = p[np.argsort(p)], w[np.argsort(p)]
        passed = p_sorted <= (np.cumsum(w_sorted) / np.sum(w_sorted) * sig_threshold)
        cutoff = p_sorted[np.max(np.where(passed))] if np.any(passed) else .0
        pvalues_fdr = xr.DataArray(np.where(np.isfinite(p_values), p_values, np.nan), dims=p_values.dims, coords=p_values.coords)
        slopes_sig = slopes.where(pvalues_fdr <= cutoff, drop=False)

        ### Figure design
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
        plt.rc('savefig', facecolor='w')
        plt.rc('axes', edgecolor='k', facecolor='w', labelcolor='dimgray')
        plt.rc('xtick', color='dimgray')
        plt.rc('ytick', color='dimgray')

        ### Create figure
        fig = plt.figure(figsize=(5.5, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
        ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
        ax.set_xticks([])
        ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontsize=5)

        ### Plot
        cmap = cmocean.cm.balance_r
        levels = np.arange(-10, 10.1, 1)
        data_cyclic, lon_cyclic = add_cyclic_point(slopes_sig.values, coord=slopes_sig.longitude.values)
        data_cyclic = np.ma.masked_invalid(data_cyclic)
        cf = ax.contourf(lon_cyclic, slopes_sig.latitude.values, data_cyclic, levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree(), corner_mask=False, zorder=1)

        ### Features
        ax.add_feature(cfeature.LAND, facecolor='lightgray', linewidth=.5, zorder=2)
        ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=3)
        ax.coastlines(linewidth=.5, resolution='50m', zorder=4)

        ### Gridlines
        lons = np.arange(-180, 181, 60)
        lats = np.arange(0, 91, 10)
        for lon in lons:
            ax.plot([lon, lon], [0, 90], transform=ccrs.PlateCarree(), color='gray', linewidth=.5, zorder=5)
        for lat in lats:
            ax.plot(np.linspace(-180, 180, 361), np.full(361, lat), transform=ccrs.PlateCarree(), color='gray', linewidth=.5, zorder=5)

        ### Colorbar
        cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.06, shrink=.8)
        cbar.ax.tick_params(labelsize=10, colors='k', labelright=True, labelleft=False, right=True, left=False)

        ### Ticks
        cbar.ax.set_xticks(np.linspace(-10, 10, 5))
        cbar.ax.set_xticklabels(['-10', '-5', '0', '+5', '+10'])
        cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.set_xlabel(r'\textbf{Change in Sea Ice Concentration [\% decade\textsuperscript{-1}]}', fontsize=13, color='k', labelpad=8)
        cbar.outline.set_edgecolor('k')
        cbar.outline.set_linewidth(.8)

        ### Title
        plt.title(r'\textbf{SEA ICE CHANGE}', color='k', fontsize=20, y=1.05)

        ### Annotations
        season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
        plt.text(1, 1.04, rf'\textbf{{{season_prefix}{first_year}–{year_max}}}', fontsize=10, rotation='horizontal', ha='right', va='top', color='k', transform=ax.transAxes)
        annotation = r'\textbf{\textit{Shading indicates trends are significant at the 95\% confidence level following area-weighted FDR correction}}'
        plt.text(.5, -.055, annotation, fontsize=5, rotation='horizontal', ha='center', color='dimgray', alpha=1, transform=ax.transAxes, va='bottom')

        ### Ensure outline is on top
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none', edgecolor='k', linewidth=1.2, zorder=10))

        ### Save figure
        plt.subplots_adjust(top=.88, bottom=.02, left=.02, right=.98)
        plt.savefig(os.path.join(fig_dir, f'Reanalysis_Arctic_ICEFRAC_Change_{text_months}_{first_year}–{year_max}.png'), dpi=1000)
        plt.close()

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')