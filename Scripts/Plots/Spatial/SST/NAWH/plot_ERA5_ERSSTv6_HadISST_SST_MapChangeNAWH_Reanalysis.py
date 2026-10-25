"""
Plot change in North Atlantic Sea Surface Temperature (SST) from ERA5, ERSSTv6, and HadISST

Notes
-----
    Data : (ERA5) https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Data : (NOAA) https://www.ncei.noaa.gov/products/extended-reconstructed-sst
    Data : (Met Office) https://www.metoffice.gov.uk/hadobs/hadisst/data/download.html
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib.colors import ListedColormap
from scipy.stats import linregress, t as student_t
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib as mpl, matplotlib.pyplot as plt
import numpy as np
import os
import re
import time
import xarray as xr

mpl.use('Agg')  # non-interactive backend to avoid tkinter issues

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

def get_coord_name(ds, names):
    for name in names:
        if name in ds.coords:
            return name
    raise ValueError(f'None of the expected coordinates were found: {names}')

def slope_pvalue(x, y):
    fit = linregress(x, y)
    resid = y - fit.intercept - fit.slope * x
    r1 = max(np.nan_to_num(np.corrcoef(resid[:-1], resid[1:])[0, 1]), 0) if np.std(resid) > 0 else 0  # lag-1 autocorrelation of residuals
    n_eff = max(len(y) * (1 - r1) / (1 + r1), 3)  # effective sample size (Santer et al. 2000)
    return fit.slope * 10, (2 * student_t.sf(abs(fit.slope) / fit.stderr * np.sqrt((n_eff - 2) / (len(y) - 2)), n_eff - 2) if fit.stderr > 0 else fit.pvalue)

def wrap_longitude_180(ds, lon_name):
    ds = ds.assign_coords({lon_name: ((ds[lon_name] + 180) % 360) - 180}).sortby(lon_name)
    return ds

def years_from_filename(filename):
    basename = os.path.basename(filename)
    match = re.search(r'(\d{4})-(\d{4})', basename)
    if not match:
        raise ValueError(f'Could not extract year range from filename: {basename}')
    return int(match.group(1)), int(match.group(2))

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/NA/SST/NAWH/')

### Dataset filenames
data_filenames = ['ERA5_Global_1940-2025_SST_Monthly.nc', 'ERSSTv6_Global_1850-2025_SST_Monthly.nc', 'HadISST_Global_1870-2025_SST_Monthly.nc']

### Month definitions
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

print(f'\n---- North Atlantic Warming Hole (NAWH) ----\n')

### Loop through each dataset
for data_filename in data_filenames:

    ### Dataset folder
    dataset_name = data_filename.split('_')[0]
    fig_dir_dataset = os.path.join(fig_dir, dataset_name)

    ### Load dataset
    filename = os.path.join(data_dir, data_filename)
    print(f'Processing {dataset_name}\n')
    datat = xr.open_dataset(filename)['sst']

    ### Extract years from filename
    year_min, year_max = years_from_filename(filename)

    ### Identify latitude and longitude coordinate names
    lat_name, lon_name = get_coord_name(datat, ['lat', 'latitude']), get_coord_name(datat, ['lon', 'longitude'])

    ### Wrap longitude to -180 to 180 if necessary
    datat = wrap_longitude_180(datat, lon_name)

    ### Loop through each initial year option
    for start_year in range(year_min, 2001, 10):

        ### Loop through each set of months
        for sel_month, mons in months_options.items():

            ### String corrections
            text_months = sel_month.split()[1]

            ### First valid year
            first_year = start_year + 1 if text_months == 'DJF' else start_year

            ### Figure
            print(f'{dataset_name} | {first_year} | {text_months}')

            ### Process data for North Atlantic only
            SST = datat.sel(time=datat['time'].dt.month.isin(mons))
            if text_months == 'DJF':
                month, year = SST.time.dt.month, SST.time.dt.year
                winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
                SST = SST.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
                SST = SST.groupby('winter_year').mean(dim='time').sel(winter_year=slice(first_year, year_max)).rename({'winter_year': 'year'})
            else:
                SST = SST.groupby('time.year').mean(dim='time').sel(year=slice(first_year, year_max))
            SST = SST.where(SST[lat_name] > 30, drop=True).where((SST[lon_name] >= -90) & (SST[lon_name] <= 90), drop=True)

            ### Compute slope and p-value
            SST = SST.chunk({'year': -1})
            years = SST.year.values
            slopes, p_values = xr.apply_ufunc(slope_pvalue, years, SST, input_core_dims=[['year'], ['year']], output_core_dims=[[], []], vectorize=True, dask='parallelized')

            ### Area-weighted FDR correction
            sig_threshold = 2 * .05  # alpha_FDR = 0.10 controls field significance at the 95% confidence level (Wilks 2016)
            p_values = p_values.compute()
            lat_name_p = get_coord_name(p_values, ['lat', 'latitude'])
            weights = np.cos(np.deg2rad(p_values[lat_name_p].values))
            weights /= np.nanmean(weights)
            p = p_values.values.flatten()[np.isfinite(p_values.values.flatten())]
            w = np.broadcast_to(weights[:, None], p_values.shape).flatten()[np.isfinite(p_values.values.flatten())]
            p_sorted, w_sorted = p[np.argsort(p)], w[np.argsort(p)]
            passed = p_sorted <= (np.cumsum(w_sorted) / np.sum(w_sorted) * sig_threshold)
            pvalues_fdr = xr.DataArray(np.where(np.isfinite(p_values), p_values, np.nan), dims=p_values.dims, coords=p_values.coords)
            sig_neg_mask = (slopes < 0) & (pvalues_fdr <= (p_sorted[np.max(np.where(passed))] if np.any(passed) else .0))

            ### Area of statistically significant negative slopes
            valid_mask = ~SST.isnull().any(dim='year')
            numerator = sig_neg_mask.weighted(np.cos(np.deg2rad(slopes[lat_name]))).sum(dim=(lat_name, lon_name)).compute()
            denominator = valid_mask.weighted(np.cos(np.deg2rad(slopes[lat_name]))).sum(dim=(lat_name, lon_name)).compute()
            frac_sig_cooling = (numerator / denominator).values
            print(f'Area of domain with statistically significant negative slopes: {frac_sig_cooling:.2%}')

            ### Figure design
            plt.rc('text', usetex=True)
            plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
            plt.rc('savefig', facecolor='w')
            plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='dimgray')
            plt.rc('xtick', color='dimgray')
            plt.rc('ytick', color='dimgray')

            ### Stippling design
            plt.rcParams['hatch.color'] = 'silver'
            plt.rcParams['hatch.linewidth'] = .5

            ### Create figure
            fig = plt.figure(figsize=(6.5, 5.5))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=-30))
            ax.set_extent([-60, 0, 37, 72], crs=ccrs.PlateCarree())
            ax.set_xticks([])
            ax.xaxis.set_tick_params(labelsize=5)
            ax.yaxis.set_tick_params(labelsize=5)
            ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontsize=5)
            ax.coastlines()
            ax.add_feature(cfeature.LAND, facecolor='dimgray', linewidth=.5, zorder=2)
            ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)
            gridlines = ax.gridlines(draw_labels=True, linewidth=.5)
            gridlines.xlabel_style = gridlines.ylabel_style = {'size': 7}

            ### Plot
            x, y = np.meshgrid(slopes[lon_name], slopes[lat_name])
            cmap = ListedColormap(plt.get_cmap('RdBu_r', 192)(np.linspace(0, .5)))
            cf = ax.contourf(x, y, slopes, levels=np.arange(-.5, .01, .025), cmap=cmap, vmin=-.5, vmax=0, transform=ccrs.PlateCarree(), extend='both')
            cs = ax.contourf(x, y, sig_neg_mask, levels=[.5, 1.5], hatches=['...', None], colors='none', transform=ccrs.PlateCarree(), zorder=1)

            ### Colorbar
            cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.12, shrink=.8, ticks=np.arange(-1, 1.01, 1), extend='both')
            cbar.ax.tick_params(labelsize=10, color='dimgray', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
            cbar.ax.set_xticks(np.linspace(-.5, 0, 6))
            cbar.ax.set_xticklabels(['-.5', '-.4', '-.3', '-.2', '-.1', '0'])
            cbar.ax.xaxis.set_ticks_position('bottom'); cbar.ax.xaxis.set_label_position('bottom')
            cbar.ax.set_xlabel(r'\textbf{Change in Sea Surface Temperature [°C decade\textsuperscript{-1}]}', fontsize=13, color='k', labelpad=8)

            ### Title
            plt.title(r'\textbf{North Atlantic Warming Hole}', color='k', fontsize=15, y=1.08)

            ### Annotations
            season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
            plt.text(0, 1.13, rf'\textbf{{{dataset_name}}}', fontsize=9, rotation='horizontal', ha='left', va='top', color='k', transform=ax.transAxes)
            plt.text(1, 1.13, rf'\textbf{{{season_prefix}{first_year}–{year_max}}}', fontsize=9, rotation='horizontal', ha='right', va='top', color='k', transform=ax.transAxes)
            plt.text(.5, -.12, r'\textbf{\textit{Stippling indicates cooling trends are significant at the 95\% confidence level following area-weighted FDR correction}}',
                     fontsize=5, rotation='horizontal', ha='center', color='dimgray', alpha=1, transform=ax.transAxes, va='bottom')

            ### Save figure
            plt.subplots_adjust(top=.86, bottom=.03, left=.08, right=.92)
            plt.savefig(os.path.join(fig_dir_dataset, f'{dataset_name}_NA_SST_Change_{text_months}_{first_year}–{year_max}.png'), dpi=1000)
            plt.close()

            ### Progress
            print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')