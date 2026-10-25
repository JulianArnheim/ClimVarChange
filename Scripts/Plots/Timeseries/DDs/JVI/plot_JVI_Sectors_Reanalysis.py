"""
Plot Jet Variability Index (JVI) timeseries from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import xarray as xr

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')
    ax.xaxis.set_ticks_position('bottom')

def concat_load_metrics(prefix):
    datasets = []
    for start_year in range(year_min, 2030, 10):
        end_year = min(start_year + 9, year_max)
        filename = os.path.join(data_dir, f'ERA5_{prefix}_Sectors_{start_year}-{end_year}_Daily.nc')
        if os.path.exists(filename):
            datasets.append(xr.open_dataset(filename))
            print(f'Loaded: {start_year}–{end_year}')
    return xr.concat(datasets, dim='time')

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

def plot_panel(ax, year_min, year_max, JVI_data, sector_code, sector_name, y_ticks=None, x_ticks=None, full_title=False, running_mean_labels=True, season=None):

    ### Adjust axes
    adjust_spines(ax, ['left', 'bottom'])
    for key in ['top', 'right']:
        ax.spines[key].set_color('none')
    for key in ['bottom', 'left']:
        ax.spines[key].set_linewidth(2)
    ax.tick_params(axis='both', length=5.5, width=2, which='major', colors='dimgray')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('k')

    ### First valid year
    first_year = year_min + 1 if season == 'DJF' else year_min

    ### Process JVI means
    JVI_monthly = JVI_data[f'JVI_{sector_code}'].resample(time='MS').mean()
    if season == 'DJF':
        month, year = JVI_monthly.time.dt.month, JVI_monthly.time.dt.year
        winter_year = xr.where(month == 12, year + 1, xr.where((month == 1) | (month == 2), year, np.nan))
        JVI_annual = JVI_monthly.assign_coords(winter_year=('time', winter_year.data)).where(~np.isnan(winter_year), drop=True)
        JVI_annual = JVI_annual.groupby('winter_year').mean(dim='time').sel(winter_year=slice(first_year, year_max)).rename({'winter_year': 'year'})
        years = JVI_annual['year'].values
    else:
        JVI_annual = JVI_monthly.resample(time='YS').mean()
        years = JVI_annual['time.year'].values

    ### Calculate LOWESS smoothing
    lowess_frac = 30 / (years[-1] - years[0])  # fraction ~ 30 years window
    JVI_lowess = lowess(JVI_annual.values, years, frac=lowess_frac, return_sorted=False)

    ### Plot
    ax.plot(JVI_monthly['time'], JVI_monthly, color='k', linewidth=.5)
    if season == 'DJF':
        annual_time = np.array([np.datetime64(f'{int(y)}-01-01') for y in years])
    else:
        annual_time = JVI_annual['time']
    ax.plot(annual_time, JVI_annual, color='indigo', linewidth=3)
    ax.plot(annual_time, JVI_lowess, color='deeppink', linewidth=4)

    ### Ticks and domain
    end_yr = 2025
    ylim = (15, 40)
    ax.set_xlim(np.datetime64(f'{year_min}-01-01'), np.datetime64(f'{end_yr}-01-01')); ax.set_ylim([15, 40])
    if x_ticks is None:
        x_ticks = [np.datetime64(f'{y}-01-01') for y in range(year_min, end_yr + 1, 5)]
    x_labels = [str(y.astype('datetime64[Y]').item().year) if y.astype('datetime64[Y]').item().year % 10 == 0 else '' for y in x_ticks]
    ax.set_xticks(x_ticks); ax.set_xticklabels(x_labels, fontsize=12)
    if y_ticks is None:
        y_ticks = list(range(15, 41, 5))
    ax.set_yticks(y_ticks); ax.set_yticklabels([str(y) for y in y_ticks], fontsize=12)

    ### Horizontal lines
    for step, style in [(5, 'solid'), (1, 'dotted')]:
        for y in np.arange(0, 50, step):
            ax.axhline(y, color='gray', linestyle=style, linewidth=.5, zorder=1)

    ### Title and labels
    if full_title == False:
        ax.set_title(rf'\textbf{{{sector_name}}}', fontsize=20)
    ax.set_xlabel(r'\textbf{Year}', fontsize=16, labelpad=10, color='k')
    ax.set_ylabel(r'\textbf{JVI [° Latitude]}', fontsize=16, labelpad=10, color='k')

    ### Annotation
    if running_mean_labels:
        x = np.datetime64(f'{end_yr-2}-01-01')
        ax.annotate('Monthly Mean', xy=(x, ylim[1] - 2.1), color='k', fontsize=11, ha='right', va='bottom')
        ax.annotate(r'\textbf{Annual Mean}', xy=(x, ylim[1] - 3.1), color='indigo', fontsize=11, ha='right', va='bottom')
        ax.annotate(r'\textbf{30-Year LOWESS}', xy=(x, ylim[1] - 4.1), color='deeppink', fontsize=11, ha='right', va='bottom')

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/DDs/JVI/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- Jet Variability Index (JVI) Timeseries ----\n')

### Month and sector settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}
sector_options = {'NH': 'Northern Hemisphere (180°W–180°E)',
                  'AM': 'American Sector (140°W–50°W)', 'AT': 'Atlantic Sector (50°W–40°E)',
                  'AS': 'Asian Sector (40°E–130°E)', 'PA': 'Pacific Sector (130°E–140°W)'}

### Load data
JVIs = concat_load_metrics('JVI')
print(f'Concatenated JVIs\n')

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

    ### Season specification
    JVIs_season = JVIs.sel(time=JVIs['time.month'].isin(mons))

    for letters in [True, False]:

        ### Figure design
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
        plt.rc('savefig', facecolor='w')
        plt.rc('axes', edgecolor='dimgray')
        plt.rc('xtick', color='dimgray')
        plt.rc('ytick', color='dimgray')
        plt.rc('axes', labelcolor='dimgray')
        plt.rc('axes', facecolor='w')

        ### Create figure
        fig = plt.figure(figsize=(14, 12))
        ax_top_pos, ax_bottom_pos = [.08, .65, .88, .3], [[.08, .36, .41, .22], [.55, .36, .41, .22], [.08, .065, .41, .22], [.55, .065, .41, .22]]
        letters_text = ['(a)', '(b)', '(c)', '(d)', '(e)']

        ### Create axes for the top panel (Northern Hemisphere)
        ax_top = fig.add_axes(ax_top_pos)
        plot_panel(ax_top, year_min, year_max, JVIs_season, sector_code='NH', sector_name='NH', running_mean_labels=True, season=text_months, full_title=True)
        ax_top.set_xlabel('')

        ### Create axes for the bottom panels (sectors, 2 x 2)
        bottom_sector_codes = ['AM', 'AT', 'AS', 'PA']
        axes = [ax_top]
        for i, (pos, sec_code) in enumerate(zip(ax_bottom_pos, bottom_sector_codes)):
            ax = fig.add_axes(pos)
            plot_panel(ax, year_min, year_max, JVIs_season, sector_code=sec_code, sector_name=sector_options[sec_code], running_mean_labels=False, season=text_months)
            axes.append(ax)
            ax.set_xlabel(r'\textbf{Year}', fontsize=18, labelpad=10, color='k') if i >= 2 else ax.set_xlabel('')
            ax.set_ylabel(r'\textbf{JVI [° Latitude]}', fontsize=18, labelpad=10, color='k') if i % 2 == 0 else ax.set_ylabel('')

        ### Title
        ax_top.set_title(rf'\textbf{{{title_season}Jet Variability Index (JVI)}}', color='k', fontsize=36, y=1.01)
        ax_top.text(.5, .985, r'\textbf{Northern Hemisphere (180°W–180°E)}', fontsize=20, color='k', ha='center', va='top', transform=ax_top.transAxes)

        ### Save figure
        if letters:
            for i, ax in enumerate(axes):
                ax.text(-.042 if i == 0 else -.09, 1.05, rf'\textbf{{{letters_text[i]}}}', ha='left', va='bottom', color='k', fontsize=18, transform=ax.transAxes)
        suffix = '_letters' if letters else ''
        fig_path = os.path.join(fig_dir, f'Reanalysis_JVI_{text_months}_{year_min}–{year_max}{suffix}.png')
        plt.savefig(fig_path, dpi=1000)
        plt.close(fig)

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')