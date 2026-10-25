"""
Plot Jet Variability Index (JVI) and North Atlantic Oscillation (NAO) timeseries from ERA5 reanalysis and NOAA CPC

Notes
-----
    Data : (ERA5) https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Data : (NOAA) https://ftp.cpc.ncep.noaa.gov/wd52dg/data/indices/nao_index.tim
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib.ticker import FixedLocator, MultipleLocator
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import os
import statsmodels.api as sm
import time
import xarray as xr

def adjust_spines_double_y(ax, spines):
    for loc, spine in ax.spines.items():
        spine.set_position(('outward', 5)) if loc in spines else spine.set_color('none')
    ax.yaxis.set_ticks_position('left' if 'left' in spines else 'right' if 'right' in spines else 'none')
    ax.xaxis.set_ticks_position('bottom' if 'bottom' in spines else 'none')
    if 'right' not in spines and 'left' not in spines:
        ax.yaxis.set_ticks([])
    if 'bottom' not in spines:
        ax.xaxis.set_ticks([])

def form_sign(x, _):
    if abs(x) < 1e-10:
        return '0'
    return f'{x:+.1f}'

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/DDs/JVI/')

### Years
year_min, year_max = 1950, 2025

print(f'\n---- Jet Variability Index (JVI) & North Atlantic Oscillation (NAO) Timeseries ----\n')

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Load JVI data
JVI_datasets = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_JVI_Sectors_{start_year}-{end_year}_Daily.nc')
    if os.path.exists(filename):
        JVI_datasets.append(xr.open_dataset(filename))
        print(f'Loaded: {start_year}–{end_year}')
JVIs = xr.concat(JVI_datasets, dim='time')
print(f'Concatenated JVIs\n')

### Load NAO data
NAOs = xr.open_dataset(os.path.join(data_dir, f'CPC_NAO_EA_1950-{year_max}_Monthly.nc'))

### LOWESS smoothing settings
lowess_num = 30  # years

### Loop through each season
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### First valid year
    first_year = year_min + 1 if text_months == 'DJF' else year_min

    ### Figure
    print(f'{text_months}')

    ### For title only, make ANNUAL blank
    title_season = '' if text_months == 'ANNUAL' else f'{text_months} '

    ### Process data
    JVI_data = JVIs.sel(time=slice(f'{year_min}-01-01', f'{year_max}-12-31')).where(JVIs.time.dt.month.isin(mons), drop=True)
    NAO_data = NAOs.sel(time=slice(f'{year_min}-01-01', f'{year_max}-12-31')).where(NAOs.time.dt.month.isin(mons), drop=True)

    ### Compute annual means over common years
    if text_months == 'DJF':
        annual = {}
        for name, data in [('JVI', JVI_data), ('NAO', NAO_data)]:
            month, year = data.time.dt.month, data.time.dt.year
            winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
            annual[name] = data.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
            annual[name] = annual[name].groupby('winter_year').mean(dim='time').sel(winter_year=slice(first_year, year_max)).rename({'winter_year': 'year'})
        JVI_annual, NAO_annual = annual['JVI'], annual['NAO']
    else:
        JVI_annual = JVI_data.groupby('time.year').mean(dim='time').sel(year=slice(year_min, year_max))
        NAO_annual = NAO_data.groupby('time.year').mean(dim='time').sel(year=slice(year_min, year_max))

    ### Common years
    common_years = np.intersect1d(JVI_annual['year'], NAO_annual['year'])
    JVI_annual, NAO_annual = JVI_annual.sel(year=common_years), NAO_annual.sel(year=common_years)
    x_years = common_years

    ### Figure design
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='w')
    plt.rc('axes', edgecolor='dimgray')
    plt.rc('xtick', labelsize=12, labelcolor='dimgray')
    plt.rc('ytick', labelsize=12, labelcolor='dimgray')
    plt.rc('axes', labelcolor='dimgray')
    plt.rc('axes', facecolor='w')

    ### Create figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # double y axis

    ### Adjust axes (1)
    adjust_spines_double_y(ax1, ['left', 'bottom'])
    ax1.spines['top'].set_color('none')
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['left'].set_linewidth(2)
    ax1.spines['right'].set_linewidth(2)
    ax1.tick_params(axis='both', length=5.5, width=2, which='major', colors='dimgray', labelcolor='dimgray', labelsize=12)

    ### Adjust axes (2)
    adjust_spines_double_y(ax2, ['right', 'bottom'])
    ax2.spines['top'].set_color('none')
    ax2.spines['right'].set_linewidth(2)
    ax2.spines['left'].set_color('none')
    ax2.spines['bottom'].set_color('none')
    ax2.tick_params(axis='both', length=5.5, width=2, which='major', colors='dimgray', labelcolor='dimgray', labelsize=12)

    ### JVI range
    if text_months == 'ANNUAL':
        y_ticks = np.arange(28, 34, 1)
        ax1.set_ylim(28, 33)
    elif text_months == 'DJF':
        y_ticks = np.arange(27, 38, 1)
        ax1.set_ylim(28, 37)
    elif text_months == 'JJA':
        y_ticks = np.arange(20, 33, 1)
        ax1.set_ylim(20, 32)

    ### Domain
    x_ticks = np.arange(1950, 2031, 5)
    ax1.set_xticks(x_ticks); ax1.set_yticks(y_ticks)
    ax1.set_xlim([year_min, 2025])

    ### Plot JVI
    ax1.plot(x_years, JVI_annual['JVI_AT'], color='k', label='JVI', linewidth=1)
    ax1.tick_params(axis='x', length=5.5, width=2, which='major', colors='dimgray', labelcolor='k', labelsize=12)
    ax1.tick_params(axis='y', length=5.5, width=2, which='major', colors='dimgray', labelcolor='k', labelsize=12)
    ax1.xaxis.set_major_locator(FixedLocator(x_ticks)); ax1.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax1.set_xticklabels([str(tick) for tick in x_ticks]); ax1.set_yticklabels([str(tick) for tick in y_ticks])
    ax1.set_xlabel(r'\textbf{Year}', fontsize=16, color='k', labelpad=10); ax1.set_ylabel(r'\textbf{JVI [° Latitude]}', fontsize=16, color='k', labelpad=10)

    ### LOWESS smoothed JVI
    lowess_smoothed_JVI = sm.nonparametric.lowess(JVI_annual['JVI_AT'], x_years, frac=lowess_num / len(x_years))
    ax1.plot(lowess_smoothed_JVI[:, 0], lowess_smoothed_JVI[:, 1], color='k', linewidth=3)

    ### Plot NAO
    ax2.plot(x_years, NAO_annual['NAO'], color='g', label='NAO', linewidth=1)
    ax2.set_ylim([-2.5, 2.5])
    ax2.yaxis.set_major_locator(MultipleLocator(.5)); ax2.yaxis.set_major_formatter(plt.FuncFormatter(form_sign))
    ax2.tick_params(axis='y', length=5.5, width=2, which='major', colors='dimgray', labelcolor='g', labelsize=12)
    ax2.set_ylabel(r'\textbf{NAO}', fontsize=16, color='g', labelpad=20)
    ax2.yaxis.label.set_rotation(270)

    ### LOWESS smoothed NAO
    lowess_smoothed_NAO = sm.nonparametric.lowess(NAO_annual['NAO'], x_years, frac=lowess_num / len(x_years))
    ax2.plot(lowess_smoothed_NAO[:, 0], lowess_smoothed_NAO[:, 1], color='g', linewidth=3)

    ### Correlations
    corr_unsmoothed, p_unsmoothed = pearsonr(JVI_annual['JVI_AT'].values, NAO_annual['NAO'].values)
    corr_smoothed = np.corrcoef(lowess_smoothed_JVI[:, 1], lowess_smoothed_NAO[:, 1])[0, 1]  # no p-value, since smoothed points are not independent
    print(f'Correlation between unsmoothed JVI and NAO: {corr_unsmoothed:+.2f} (p = {p_unsmoothed:{".2f" if p_unsmoothed >= .01 else ".2e"}})')
    print(f'Correlation between LOWESS smoothed JVI and NAO: {corr_smoothed:+.2f}')

    ### Title and annotations
    plt.title(rf'\textbf{{Jet Variability Index (JVI): {title_season}Atlantic Sector (50°W–40°E)}}', color='k', fontsize=20, y=1)
    ax1.text(.88, .94, f'Smoothed Correlation: {corr_smoothed:+.2f}', transform=ax1.transAxes, ha='right', va='top', color='k', fontsize=13)
    ax1.text(.12, .94, f'{lowess_num}-Year LOWESS Smoothing', transform=ax1.transAxes, ha='left', va='top', color='k', fontsize=13)

    ### Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_AT_JVI_NAO_{text_months}_{year_min}–{year_max}.png'), dpi=1000)
    plt.close(fig)

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')