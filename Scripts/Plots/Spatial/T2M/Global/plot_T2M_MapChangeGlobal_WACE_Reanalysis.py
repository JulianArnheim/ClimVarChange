"""
Plot change in 2-m Air Temperature (T2M) from ERA5 reanalysis over the period 1998–2012, as used in Outten et al. (2023)

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from cartopy.util import add_cyclic_point
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
fig_dir = os.path.join(base_dir, 'Figures/Spatial/Global/T2M/Slope/')

### Years
year_min, year_max = 1998, 2012

print(f'\n---- Warm Arctic–Cold Eurasia (T2M) ----\n')

### Load or compute slope and p-value
files = {'T2M_slope': os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_T2M_Slopes_DJF.nc'),
         'T2M_pval': os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_T2M_PValues_DJF.nc'),
         'SLP_slope': os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_SLP_Slopes_DJF.nc'),
         'SLP_pval': os.path.join(data_dir, f'ERA5_Global_{year_min}-{year_max}_SLP_PValues_DJF.nc')}

if all(os.path.exists(files[key]) for key in files):
    T2M_slopes, T2M_pvalues = (xr.open_dataset(files['T2M_slope'])['T2M_slopes'], xr.open_dataset(files['T2M_pval'])['T2M_pvalues'])
    SLP_slopes, SLP_pvalues = (xr.open_dataset(files['SLP_slope'])['SLP_slopes'], xr.open_dataset(files['SLP_pval'])['SLP_pvalues'])
    print(f'Using Previously Computed ERA5 T2M and SLP Slopes\n')

else:
    print(f'Files being made:')
    for f in files.values():
        print(f'  - {f}')

    ### Load preprocessed monthly T2M [°C] and SLP [hPa]
    datat = xr.open_dataset(os.path.join(data_dir, 'ERA5_Global_1940-2025_T2M_Monthly.nc'))['t2m']
    datap = xr.open_dataset(os.path.join(data_dir, 'ERA5_Global_1940-2025_SLP_Monthly.nc'))['SLP']

    ### Process data (DJF seasons: 1998–2012 → Dec 1997 to Feb 2012)
    T2M_season = datat.sel(time=datat['time'].dt.month.isin([12, 1, 2]))
    T2M_season = T2M_season.groupby(xr.where(T2M_season['time'].dt.month == 12, T2M_season['time'].dt.year + 1, T2M_season['time'].dt.year))
    T2M = T2M_season.mean(dim='time').rename({'group': 'year'}).sel(year=slice(year_min, year_max))
    SLP_season = datap.sel(time=datap['time'].dt.month.isin([12, 1, 2]))
    SLP_season = SLP_season.groupby(xr.where(SLP_season['time'].dt.month == 12, SLP_season['time'].dt.year + 1, SLP_season['time'].dt.year))
    SLP = SLP_season.mean(dim='time').rename({'group': 'year'}).sel(year=slice(year_min, year_max))
    time_index = T2M.year.values

    ### Parallel computation of slopes and p-values
    ufunc_kwargs = dict(input_core_dims=[['year'], ['year']], output_core_dims=[[], []], vectorize=True, dask='parallelized', output_dtypes=[float, float])
    T2M_slopes, T2M_pvalues = xr.apply_ufunc(slope_pvalue, time_index, T2M, **ufunc_kwargs)
    SLP_slopes, SLP_pvalues = xr.apply_ufunc(slope_pvalue, time_index, SLP, **ufunc_kwargs)
    T2M_slopes, T2M_pvalues, SLP_slopes, SLP_pvalues = T2M_slopes.compute(), T2M_pvalues.compute(), SLP_slopes.compute(), SLP_pvalues.compute()

    ### Save
    T2M_slopes.to_dataset(name='T2M_slopes').to_netcdf(files['T2M_slope'])
    T2M_pvalues.to_dataset(name='T2M_pvalues').to_netcdf(files['T2M_pval'])
    SLP_slopes.to_dataset(name='SLP_slopes').to_netcdf(files['SLP_slope'])
    SLP_pvalues.to_dataset(name='SLP_pvalues').to_netcdf(files['SLP_pval'])
    print(f'Saved {", ".join(files.values())}, the ERA5 T2M and SLP slopes and p-values\n')

### Significance mask
sig_mask = T2M_pvalues < .05

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='dimgray')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(central_longitude=60, central_latitude=70))
ax.set_xticks([]); ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels(), fontsize=5)
ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor='none', linewidth=.5, zorder=2)
ax.add_feature(cfeature.LAKES, edgecolor='k', facecolor='none', linewidth=.5, zorder=2)

### Gridlines
for lon0 in np.arange(-180, 181, 60):
    ax.plot(np.full(181, lon0), np.linspace(-90, 90, 181), transform=ccrs.PlateCarree(), color='silver', linewidth=.5, zorder=10)
for lat0 in np.arange(-90, 91, 30):
    ax.plot(np.linspace(-180, 180, 361), np.full(361, lat0), transform=ccrs.PlateCarree(), color='silver', linewidth=.5, zorder=10)

### Add cyclic point
lon = T2M_slopes.longitude
lat = T2M_slopes.latitude
T2M_cyc, lon_cyc = add_cyclic_point(T2M_slopes, coord=lon)
sig_cyc, _ = add_cyclic_point(sig_mask, coord=lon)
SLP_cyc, _ = add_cyclic_point(SLP_slopes, coord=lon)

### Plot T2M trend (shading)
cf = ax.contourf(lon_cyc, lat, T2M_cyc, levels=np.arange(-4, 4.1, .5), cmap=plt.get_cmap('RdBu_r', lut=192), vmin=-4, vmax=4, transform=ccrs.PlateCarree(), extend='both')
ax.contourf(lon_cyc, lat, sig_cyc, levels=[.5, 1.5], hatches=['....'], colors='none', transform=ccrs.PlateCarree(), zorder=3)

### Plot SLP trend (contours)
cs = ax.contour(lon_cyc, lat, SLP_cyc, levels=np.setdiff1d(np.arange(-30, 33, 3), 0), colors='#33D816', linewidths=1, transform=ccrs.PlateCarree(), zorder=5)
texts = ax.clabel(cs, fmt=lambda x: rf'\textbf{{{x:+.0f}}}', fontsize=7, inline=True, use_clabeltext=True)

### Colorbar
cbar = plt.colorbar(cf, orientation='horizontal', aspect=40, pad=.08, shrink=.8, extend='both')
cbar.ax.tick_params(labelsize=10, color='dimgray', labelcolor='k', labelright=True, labelleft=False, right=True, left=False)
cbar.ax.set_xticks(np.linspace(-4, 4, 9))
cbar.ax.set_xticklabels(['-4', '-3', '-2', '-1', '0', '+1', '+2', '+3', '+4'])
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.ax.xaxis.set_label_position('bottom')
cbar.ax.set_xlabel(r'\textbf{Change in 2-m Air Temperature [°C decade\textsuperscript{-1}] and Sea Level Pressure [hPa decade\textsuperscript{-1}]}', fontsize=8, color='k', labelpad=4)

### Title
plt.title(r"\textbf{``WARM ARCTIC–COLD EURASIA''}", color='k', fontsize=20, y=1.01)

### Annotations
plt.text(1.15, .96, fr'\textbf{{DJF {year_min}–{year_max}}}', fontsize=9, rotation='horizontal', ha='right', va='top', color='k', transform=ax.transAxes)
plt.text(.5, -.07, r'\textbf{\textit{Stippling indicates trends are significant at the 95\% confidence level}}', fontsize=5, ha='center', color='k', alpha=1, transform=ax.transAxes, va='bottom')

### Save figure
plt.subplots_adjust(top=.9, bottom=0, left=.02, right=.98)
plt.savefig(os.path.join(fig_dir, f'Reanalysis_Global_T2M_Change_WACE_DJF_{year_min}–{year_max}.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')