"""
Plot change in Sea Surface Temperature (SST) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from scipy.ndimage import binary_dilation, label
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

def patch_small_nan_clusters(data, max_size=20, min_perimeter_fraction=2/3, dilation_iters=2):
    arr = data.values.copy()
    labels, nlabels = label(np.isnan(arr))
    for i in range(1, nlabels + 1):
        cluster = labels == i
        if cluster.sum() > max_size:
            continue
        ring = binary_dilation(cluster, iterations=dilation_iters)
        ring &= ~cluster
        total_ring = ring.sum()
        if total_ring == 0:
            continue
        valid_ring = np.isfinite(arr[ring])
        if valid_ring.sum() / total_ring < min_perimeter_fraction:
            continue
        values = arr[ring][valid_ring]
        arr[cluster] = np.median(values)
    return xr.DataArray(arr, coords=data.coords, dims=data.dims, attrs=data.attrs)

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
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Global/SST/Slope/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Change in Sea Surface Temperature (SST) ----\n')

### Load data
datat = xr.open_dataset(os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_SST_Monthly.nc'))['sst']

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
        SST = datat.sel(time=datat['time'].dt.month.isin(mons))
        if text_months == 'DJF':
            month, year = SST.time.dt.month, SST.time.dt.year
            winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
            SST = SST.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
            SST = SST.groupby('winter_year').mean(dim='time').sel(winter_year=slice(first_year, year_max)).rename({'winter_year': 'year'})
        else:
            SST = SST.groupby('time.year').mean(dim='time').sel(year=slice(first_year, year_max))

        ### Compute slope and p-value
        SST = SST.chunk({'year': -1})  # year rechunking
        time_index = SST.year.values
        slopes, p_values = xr.apply_ufunc(slope_pvalue, time_index, SST, input_core_dims=[['year'], ['year']], output_core_dims=[[], []], vectorize=True, dask='parallelized')

        ### Area-weighted FDR correction
        sig_threshold = 2 * .05  # alpha_FDR = 0.10 controls field significance at the 95% confidence level (Wilks 2016)
        p_values = p_values.compute()
        weights = np.cos(np.deg2rad(p_values['latitude'].values))
        weights /= np.nanmean(weights)
        p, w = p_values.values.flatten(), np.broadcast_to(weights[:, None], p_values.shape).flatten()
        p, w = p[np.isfinite(p)], w[np.isfinite(p)]
        p_sorted, w_sorted = p[np.argsort(p)], w[np.argsort(p)]
        passed = p_sorted <= (np.cumsum(w_sorted) / np.sum(w_sorted) * sig_threshold)
        cutoff = p_sorted[np.max(np.where(passed))] if np.any(passed) else .0
        valid_pvalues = np.where(np.isfinite(p_values), p_values, np.nan)
        pvalues_fdr = xr.DataArray(valid_pvalues, dims=p_values.dims, coords=p_values.coords)
        insig_mask = (pvalues_fdr > cutoff).where(np.isfinite(pvalues_fdr))  # NaN over land

        ### Bring into memory
        slopes, pvalues_fdr = slopes.compute(), pvalues_fdr.compute()

        ### Sea surface temperature change statistics (°C decade⁻¹) and area without statistical significance
        weights = np.cos(np.deg2rad(slopes.latitude))
        Mean = slopes.weighted(weights).mean(dim=('latitude', 'longitude')).values
        Min, Max = slopes.min().values, slopes.max().values
        print(f'Global [Min/Mean/Max]: {Min:+.4f} / {Mean:+.4f} / {Max:+.4f} °C decade⁻¹')
        frac_insignificant = (insig_mask).weighted(weights).mean(dim=('latitude', 'longitude')).values
        print(f'Ocean area without statistical significance: {frac_insignificant:.2%}')

        ### Patch islands after the statistics (display only)
        slopes, insig_mask = patch_small_nan_clusters(slopes), patch_small_nan_clusters(insig_mask)

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
        ax.add_feature(cfeature.LAND, facecolor='dimgray', linewidth=.5, zorder=2); ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
        gridlines = ax.gridlines(draw_labels=True, linewidth=.5, zorder=4)
        gridlines.xlabel_style = gridlines.ylabel_style = {'size': 7}

        ### Plot
        x, y = np.meshgrid(datat.longitude, datat.latitude)
        cf = ax.contourf(x, y, slopes, levels=np.arange(-1, 1.1, .1), cmap=plt.get_cmap('RdBu_r', lut=192), vmin=-1, vmax=1, transform=ccrs.PlateCarree(), extend='both')
        ax.contourf(x, y, insig_mask, levels=[.5, 1.5], colors='w', transform=ccrs.PlateCarree(), zorder=1)  # uncertainty mask

        ### Colorbar
        cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.1, shrink=.8, extend='both')
        cbar.ax.tick_params(labelsize=10, color='dimgray', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
        cbar.ax.set_xticks(np.linspace(-1, 1, 11)); cbar.ax.set_xticklabels(['-1', '-.8', '-.6', '-.4', '-.2', '0', '+.2', '+.4', '+.6', '+.8', '+1'])
        cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
        cbar.ax.set_xlabel(r'\textbf{Change in Sea Surface Temperature [°C decade\textsuperscript{-1}]}', fontsize=13, color='k', labelpad=8)

        ### Title
        plt.title(r'\textbf{SEA SURFACE TEMPERATURE CHANGE}', color='k', fontsize=20, y=1.12)

        ### Annotations
        plt.text(.88, 1.08, rf'\textbf{{{"" if text_months == "ANNUAL" else f"{text_months} "}{first_year}–{year_max}}}', fontsize=10, ha='left', va='top', color='k', transform=ax.transAxes)
        plt.text(.5, -.13, r'\textbf{\textit{Shading indicates trends are significant at the 95\% confidence level following area-weighted FDR correction}}',
                 fontsize=5, rotation='horizontal', ha='center', color='dimgray', alpha=1, transform=ax.transAxes, va='bottom')

        ### Save figure
        plt.subplots_adjust(top=.98, bottom=.03, left=.08, right=.92)
        plt.savefig(os.path.join(fig_dir, f'Reanalysis_Global_SST_Change_{text_months}_{first_year}–{year_max}.png'), dpi=1000)
        plt.close()

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')