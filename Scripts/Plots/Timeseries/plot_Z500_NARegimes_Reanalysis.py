"""
Plot North Atlantic (NA) wintertime daily weather regimes of
500 hPa geopotential height (Z500) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib.ticker import FixedFormatter
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

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/Z500/')

### Years
year_min, year_max = 1940, 2025

print(f'\n---- North Atlantic Weather Regimes (Z500) ----\n')

### Winter seasons, matching Analysis/calc_NARegimes_1940–2025_Reanalysis.py
month_options = {'DJF': [12, 1, 2], 'DJFM': [12, 1, 2, 3]}

### Loop through seasons
for text_months, months in month_options.items():
    print(f'{text_months}')

    ### Load the daily regime labels computed by calc_NARegimes (sqrt(cos) weighting, 3-day persistence minimum)
    cluster_filename = os.path.join(data_dir, f'ERA5_NA_Z500_Clustered_Regimes_{text_months}_{year_min + 1}-{year_max}.nc')
    if not os.path.exists(cluster_filename):
        raise FileNotFoundError(f'Run Analysis/calc_NARegimes_1940–2025_Reanalysis.py first: {cluster_filename} not found')
    cluster_data = xr.open_dataset(cluster_filename)
    cluster_order = [s.strip() for s in cluster_data.attrs['cluster_order'].split(',')]
    labels, n_clusters = cluster_data['labels'], len(cluster_order)
    print(f'Loaded: {cluster_filename}')

    ### Fraction of days in each regime per winter, labelled by the January year
    labels = labels.assign_coords(winter_year=('time', xr.where(labels.time.dt.month == 12, labels.time.dt.year + 1, labels.time.dt.year).data))
    years = np.unique(labels['winter_year'].values)
    cluster_percent_per_year = np.zeros((years.size, n_clusters))
    for i, year in enumerate(years):
        year_labels = labels.where(labels['winter_year'] == year, drop=True).values
        cluster_percent_per_year[i] = [np.mean(year_labels == j) for j in range(n_clusters)]

    ### Figure design
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='w')
    plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='dimgray')
    plt.rc('xtick', color='dimgray')
    plt.rc('ytick', color='dimgray')

    ### Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    ### Adjust axes
    adjust_spines(ax, ['left', 'bottom'])
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

    ### Horizontal lines
    for i in np.arange(0, 1.5, .1):
        plt.axhline(i, color='gray', linewidth=1 if i == 0 else .5, linestyle='-' if i == 0 else 'dotted')

    ### Figure settings
    regime_info = {'NAO+': ('Positive NAO (NAO+)', 'red'), 'BL': ('European Blocking (BL)', 'royalblue'), 'NAO-': ('Negative NAO (NAO-)', 'green'), 'AR': ('Atlantic Ridge (AR)', 'purple')}
    titles, outline_colors = [regime_info[c][0] for c in cluster_order], [regime_info[c][1] for c in cluster_order]

    ### Plot
    for j in range(n_clusters):
        plt.plot(years, cluster_percent_per_year[:, j], label=titles[j], color=outline_colors[j], linewidth=1)
    lowess_num = 30
    for j in range(n_clusters):
        smoothed = lowess(cluster_percent_per_year[:, j], years, frac=lowess_num / len(years))
        plt.plot(smoothed[:, 0], smoothed[:, 1], color=outline_colors[j], linewidth=3, alpha=1)

    ### Ticks and domain
    xtick_vals, ytick_vals = np.arange(1940, 2026, 5), np.arange(0, 1.1, .1)
    plt.xticks(xtick_vals, xtick_vals, fontsize=12, color='k'); plt.yticks(ytick_vals, ytick_vals, fontsize=12, color='k')
    plt.gca().yaxis.set_major_formatter(FixedFormatter([f'{tick:.1g}'.replace('0.', '.') if tick < 1 else f'{tick:.1g}' for tick in ytick_vals]))
    plt.xlim([year_min, 2025]); plt.ylim([0, .8])

    ### Title, labels, annotation, and legend
    plt.title(rf'\textbf{{{text_months} Euro-Atlantic Weather Regimes}}', color='k', fontsize=25, y=1.01)
    plt.xlabel(rf'\textbf{{Year}}', fontsize=16, color='k', labelpad=10); plt.ylabel(rf'\textbf{{Frequency}}', fontsize=16, color='k', labelpad=10)
    plt.annotate(rf'\textbf{{{lowess_num}-Year LOWESS Smoothing}}', xy=(1982, .75), color='k', fontsize=13, ha='center', va='bottom')
    l = plt.legend(shadow=False, fontsize=12, loc='upper left', bbox_to_anchor=(0, 1), fancybox=True, ncol=1, frameon=False)
    for text in l.get_texts():
        text.set_color('k')

    ### Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_NA_Z500_Regimes_{text_months}_{year_min + 1}–{year_max}.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')