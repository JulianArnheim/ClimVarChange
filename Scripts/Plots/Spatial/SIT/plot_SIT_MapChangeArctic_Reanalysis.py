"""
Plot change in Sea Ice Thickness (SIT) from the PIOMAS ice-ocean model

Notes
-----
    Data : https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/model_grid
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
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Arctic/SIT/Slope/')

### Years
year_min, year_max = 1979, 2025

print(f'\n---- Global Warming (SIT) ----\n')

### Load data
datathick = xr.open_dataset(os.path.join(data_dir, f'PIOMAS_Arctic_{year_min}-{year_max}_SIT_Monthly.nc'))['SIT']

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### Figure
    print(f'{year_min}–{year_max} | {text_months}')

    ### Process data
    SIT = datathick.sel(time=datathick['time'].dt.month.isin(mons))
    if text_months == 'DJF':
        month, year = SIT.time.dt.month, SIT.time.dt.year
        winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
        SIT = SIT.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
        SIT = SIT.groupby('winter_year').mean(dim='time').sel(winter_year=slice(year_min, year_max)).rename({'winter_year': 'year'})
    else:
        SIT = SIT.groupby('time.year').mean(dim='time').sel(year=slice(year_min, year_max))
    SIT = SIT.where(SIT.latitude >= 0, drop=True)

    ### Compute slope and p-value
    SIT = SIT.chunk({'year': -1})  # year rechunking
    slopes, p_values = xr.apply_ufunc(slope_pvalue, SIT.year.values, SIT, input_core_dims=[['year'], ['year']], output_core_dims=[[], []], vectorize=True, dask='parallelized')

    ### Area-weighted FDR correction
    sig_threshold, p_values = 2 * .05, p_values.compute()  # alpha_FDR = 0.10 controls field significance at the 95% confidence level (Wilks 2016)
    weights = np.cos(np.deg2rad(p_values['latitude'].values))
    weights /= np.nanmean(weights)
    p = p_values.values.flatten()
    w = np.broadcast_to(weights[:, None], p_values.shape).flatten()
    p, w = p[np.isfinite(p)], w[np.isfinite(p)]
    p_sorted, w_sorted = p[np.argsort(p)], w[np.argsort(p)]
    passed = p_sorted <= (np.cumsum(w_sorted) / np.sum(w_sorted) * sig_threshold)
    cutoff = p_sorted[np.max(np.where(passed))] if np.any(passed) else .0
    valid_pvalues = np.where(np.isfinite(p_values), p_values, np.nan)
    pvalues_fdr = xr.DataArray(valid_pvalues, dims=p_values.dims, coords=p_values.coords)
    insig_mask = pvalues_fdr > cutoff

    ### Bring into memory
    slopes, pvalues_fdr, insig_mask = slopes.compute(), pvalues_fdr.compute(), insig_mask.compute()

    ### Fill missing values at the pole with zonal mean of nearest latitude
    lat_max = slopes.latitude.max()
    if slopes.sel(latitude=lat_max).isnull().any():
        slopes.loc[dict(latitude=lat_max)] = slopes.sel(latitude=slopes.latitude.where(slopes.latitude < lat_max, drop=True).max()).mean(dim='longitude')

    ### Sea ice change statistics (m decade⁻¹)
    weights = np.cos(np.deg2rad(slopes.latitude))
    Min, Max = slopes.min().values, slopes.max().values
    print(f'Arctic [Min/Max]: {Min:+.4f} / {Max:+.4f} m decade⁻¹')

    ### Area without statistical significance
    insig_mask = insig_mask.where(np.isfinite(pvalues_fdr))  # NaN outside the PIOMAS ice domain
    frac_insignificant = insig_mask.weighted(weights).mean(dim=('latitude', 'longitude')).values
    print(f'Area of the ice domain without statistical significance: {frac_insignificant:.2%}')

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
    levels = np.arange(-1, 1.1, .1)
    data_cyclic, lon_cyclic = add_cyclic_point(slopes.values, coord=slopes.longitude)
    x_wrap, y_wrap = np.meshgrid(lon_cyclic, slopes.latitude.values)
    cf = ax.contourf(x_wrap, y_wrap, data_cyclic, levels=levels, cmap=cmap, vmin=-1, vmax=1, transform=ccrs.PlateCarree(), extend='both')
    mask_cyclic, _ = add_cyclic_point(insig_mask.fillna(0).values, coord=slopes.longitude)
    ax.contourf(x_wrap, y_wrap, mask_cyclic, levels=[.5, 1.5], colors='w', transform=ccrs.PlateCarree(), zorder=1)
    ax.add_feature(cfeature.LAND, edgecolor='k', facecolor='lightgray', linewidth=.5, zorder=1)
    ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
    ax.coastlines(linewidth=.5, resolution='50m', zorder=3)

    ### Gridlines
    lons, lats = np.arange(-180, 181, 60), np.arange(0, 91, 10)
    for lon in lons:
        ax.plot([lon, lon], [0, 90], transform=ccrs.PlateCarree(), color='gray', linewidth=.5, zorder=5)
    for lat in lats:
        ax.plot(np.linspace(-180, 180, 361), np.full(361, lat), transform=ccrs.PlateCarree(), color='gray', linewidth=.5, zorder=5)

    ### Colorbar
    cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.06, shrink=.8, extend='neither')
    cbar.ax.tick_params(labelsize=10, colors='k', labelright=True, labelleft=False, right=True, left=False)
    cbar.ax.set_xticks(np.linspace(-1, 1, 5))
    cbar.ax.set_xticklabels(['-1', '-.5', '0', '+.5', '+1'])
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.set_xlabel(r'\textbf{Change in Sea Ice Thickness [m decade\textsuperscript{-1}]}', fontsize=13, color='k', labelpad=8)
    cbar.outline.set_edgecolor('k'); cbar.outline.set_linewidth(.8)

    ### Title
    plt.title(r'\textbf{SEA ICE CHANGE}', color='k', fontsize=20, y=1.05)

    ### Annotations
    season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
    plt.text(1, 1.04, rf'\textbf{{{season_prefix}{year_min}–{year_max}}}', fontsize=10, rotation='horizontal', ha='right', va='top', color='k', transform=ax.transAxes)
    annotation = r'\textbf{\textit{Shading indicates trends are significant at the 95\% confidence level following area-weighted FDR correction}}'
    plt.text(.5, -.055, annotation, fontsize=5, rotation='horizontal', ha='center', color='dimgray', alpha=1, transform=ax.transAxes, va='bottom')

    ### Ensure outline is on top
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor='none', edgecolor='k', linewidth=1.2, zorder=10))

    ### Save figure
    plt.subplots_adjust(top=.88, bottom=.02, left=.02, right=.98)
    plt.savefig(os.path.join(fig_dir, f'PIOMAS_Arctic_SIT_Change_{text_months}_{year_min}–{year_max}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')