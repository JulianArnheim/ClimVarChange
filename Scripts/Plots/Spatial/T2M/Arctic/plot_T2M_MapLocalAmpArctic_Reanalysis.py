"""
Plot change in 2-m Air Temperature (T2M) relative to the rate of global warming
(i.e. local Arctic amplification) from Sweeney et al. (2023) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from cartopy.util import add_cyclic_point
from matplotlib.colors import LinearSegmentedColormap
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
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Arctic/T2M/Local AA/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Local Arctic Amplification (T2M) ----\n')

### Load data
datat = xr.open_dataset(os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_T2M_Monthly.nc'))['t2m']

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each initial year option
for start_year in range(1960, 1981, 10):

    ### Loop through each set of months
    for sel_month, mons in months_options.items():

        ### String corrections
        text_months = sel_month.split()[1]

        ### First valid year
        first_year = start_year + 1 if text_months == 'DJF' else start_year

        ### Figure
        print(f'{first_year} | {text_months}')

        ### Process data
        T2M = datat.sel(time=datat['time'].dt.month.isin(mons))
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
        sig_threshold, p_values = 2 * .05, p_values.compute()  # alpha_FDR = 0.10 controls field significance at the 95% confidence level (Wilks 2016)
        weights = np.cos(np.deg2rad(p_values['latitude'].values))
        weights /= np.nanmean(weights)
        p, w = p_values.values.flatten(), np.broadcast_to(weights[:, None], p_values.shape).flatten()
        p, w = p[np.isfinite(p)], w[np.isfinite(p)]
        p_sorted, w_sorted = p[np.argsort(p)], w[np.argsort(p)]
        passed = p_sorted <= (np.cumsum(w_sorted) / np.sum(w_sorted) * sig_threshold)
        cutoff = p_sorted[np.max(np.where(passed))] if np.any(passed) else .0
        pvalues_fdr = xr.DataArray(np.where(np.isfinite(p_values), p_values, np.nan), dims=p_values.dims, coords=p_values.coords)
        insig_mask = pvalues_fdr > cutoff

        ### Spatial mean warming rate (°C decade⁻¹) and area without statistical significance
        weights = np.cos(np.deg2rad(slopes.latitude))
        mean_warming = slopes.weighted(weights).mean(dim=('latitude', 'longitude'))
        frac_insignificant = insig_mask.weighted(weights).mean(dim=('latitude', 'longitude')).values
        print(f'Global area without statistical significance: {frac_insignificant:.2%}')

        ### Compute local amplification and Arctic mean AA >70°N
        AA = slopes / mean_warming
        AA_min, AA_mean, AA_max = (float(AA.min()), float(AA.weighted(weights).mean(dim=('latitude', 'longitude'))), float(AA.max()))
        print(f'Relative Warming [Min/Mean/Max]: {AA_min:.2f} / {AA_mean:.2f} / {AA_max:.2f}')
        AA_Arctic = AA.where(AA.latitude >= 70, drop=True)
        AA_Arctic_mean = float(AA_Arctic.weighted(np.cos(np.deg2rad(AA_Arctic.latitude))).mean(dim=('latitude', 'longitude')))
        print(f'Arctic Amplification ≥70°N: {AA_Arctic_mean:.2f}')

        ### Figure design
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
        plt.rc('savefig', facecolor='w')
        plt.rc('axes', edgecolor='k', facecolor='w')
        plt.rc('xtick', color='dimgray')
        plt.rc('ytick', color='dimgray')

        ### Create figure
        fig = plt.figure(figsize=(5.5, 5))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
        ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())
        ax.coastlines(linewidth=.5, resolution='50m')
        ax.set_xticks([])
        ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontsize=5)
        ax.add_feature(cfeature.LAND, facecolor='none', linewidth=.5, zorder=2)
        ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
        gridlines = ax.gridlines(draw_labels=False, linewidth=.5, zorder=4)
        gridlines.top_labels = gridlines.bottom_labels = gridlines.right_labels = False
        gridlines.left_labels = True

        ### Colors
        tick_bounds = np.arange(0, 6.1, .5)
        colors = ['w', '#ffb347', '#ff5733', '#b30000ff']
        cmap_orange_red = LinearSegmentedColormap.from_list('w_or_r', colors)

        ### Plot
        AA_cyc, lon_cyc = add_cyclic_point(AA.values, coord=AA.longitude.values)
        mask_cyc, _ = add_cyclic_point(insig_mask.values, coord=AA.longitude.values)
        AA_plot = xr.DataArray(AA_cyc, dims=('latitude', 'longitude'), coords={'latitude': AA.latitude, 'longitude': lon_cyc})
        mask_plot = xr.DataArray(mask_cyc, dims=('latitude', 'longitude'), coords={'latitude': AA.latitude, 'longitude': lon_cyc})
        cf = ax.contourf(lon_cyc, AA.latitude, AA_plot, levels=tick_bounds, cmap=cmap_orange_red, transform=ccrs.PlateCarree(), extend='max')
        ax.contourf(lon_cyc, AA.latitude, mask_plot, levels=[.5, 1.5], colors='w', transform=ccrs.PlateCarree(), zorder=1)

        ### Colorbar
        cbar = plt.colorbar(cf, orientation='horizontal', aspect=36, pad=.08, shrink=.75, extend='both')
        cbar.ax.tick_params(labelsize=10, colors='k', labelright=True, labelleft=False, right=True, left=False)
        cbar.set_ticks(np.arange(0, 6.1, 1))
        cbar.ax.set_xticklabels([f'{t:d}' for t in range(0, 7)])
        cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.set_xlabel(r'\textbf{Change in 2-m Air Temperature Relative to Global Mean [\textit{x} times faster]}', fontsize=10, color='k', labelpad=8)

        ### Arctic ring
        theta = np.linspace(-180, 180, 721)
        Arctic = np.full_like(theta, 70.0)
        ax.plot(theta, Arctic, transform=ccrs.PlateCarree(), color='k', linewidth=2, zorder=5)

        ### Title
        plt.title(r'\textbf{ARCTIC AMPLIFICATION}', color='k', fontsize=22, y=1.06)

        ### Annotations
        plt.text(1, 1.04, rf'\textbf{{{"" if text_months == "ANNUAL" else f"{text_months} "}{year_min}–{year_max}}}', fontsize=10, ha='right', va='top', transform=ax.transAxes)
        plt.text(.02, .98, rf'\textbf{{AA: {AA_Arctic_mean:.2f}}}', fontsize=10, ha='left', va='top', transform=ax.transAxes, bbox=dict(boxstyle='round,pad=0.25', fc='w', ec='k', lw=2))
        plt.text(.5, -.06, r'\textbf{\textit{Shading indicates trends are significant at the 95\% confidence level following area-weighted FDR correction}}',
                 fontsize=5, ha='center', color='dimgray', transform=ax.transAxes)

        ### Save figure
        plt.subplots_adjust(top=.87, bottom=.01, left=.02, right=.98)
        plt.savefig(os.path.join(fig_dir, f'Reanalysis_Global_T2M_LocalAA_{text_months}_{first_year}–{year_max}.png'), dpi=1000)
        plt.close()

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')