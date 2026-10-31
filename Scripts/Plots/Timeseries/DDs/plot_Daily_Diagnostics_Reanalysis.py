"""
Plot the four daily diagnostic climatologies: Zonal Index (ZON), Sinuosity
Index (SIN), Blocking Index (BLO), Jet Variability Index (JVI) for reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib.ticker import FixedFormatter
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
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks_position('bottom')

def calc_climo(data, index_prefix):
    climo = data.sortby('time').groupby('time.dayofyear').mean().sel(dayofyear=slice(1, 365))
    return climo[[f"{index_prefix}_{sector.split(' ')[0]}" for sector in sector_options]]

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

def plot_panel(ax, climo, index_prefix, title, y_ticks, y_tick_labels, y_lim, hlines, x_label=None, y_label=None):

    ### Adjust axes
    adjust_spines(ax, ['left', 'bottom'])
    for key in ['top', 'right']:
        ax.spines[key].set_color('none')
    for key in ['bottom', 'left']:
        ax.spines[key].set_linewidth(2)
    ax.tick_params(axis='both', length=5.5, width=2, which='major', colors='dimgray')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('k')

    ### Horizontal lines
    for hline in hlines:
        if len(hline) == 2:
            step, style = hline
            for y in np.arange(y_lim[0], y_lim[1] + step, step):
                ax.axhline(y, color='gray', linestyle=style, linewidth=.5, zorder=1)
        else:
            y, style, _ = hline
            ax.axhline(y, color='gray', linestyle=style, linewidth=.5, zorder=1)

    ### Plot each sector
    for sector in sector_options:
        days = np.arange(1, 366) if 'month_day' in climo.coords else climo['dayofyear']
        timeseries = climo[f"{index_prefix}_{sector.split(' ')[0]}"]
        if sector.startswith('NH'):
            ax.plot(days, timeseries, color=colors[sector], label=sector, linewidth=3, zorder=5)
        else:
            ax.plot(days, timeseries, color=colors[sector], label=sector, linewidth=2, zorder=2)

    ### Title
    ax.set_title(title, color='k', fontsize=18, y=1.01)

    ### Ticks
    ax.set_xticks(month_positions, month_labels, fontsize=11, color='k', rotation=0)
    ax.set_yticks(y_ticks, y_ticks, fontsize=11, color='k')
    if y_tick_labels is not None:
        ax.yaxis.set_major_formatter(FixedFormatter(y_tick_labels))

    ### Domain
    ax.set_xlim(1, 365); ax.set_ylim(y_lim)

    ### Labels
    if y_label:
        ax.set_ylabel(y_label, fontsize=12, color='k', labelpad=10)
    if x_label:
        ax.set_xlabel(x_label, fontsize=12, color='k', labelpad=10)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/DDs/')

### Years and epochs
year_min, year_max = 1940, 2025
epochs = [(year_min, year_max), (1981, 2010)]

print(f'\n---- Daily Diagnostic Climatology ----\n')

### Load data
ZONs = concat_load_metrics('ZON')
print(f'Concatenated ZONs\n')
SINs = concat_load_metrics('SIN')
print(f'Concatenated SINs\n')
BLOs = concat_load_metrics('BLO_5d')
print(f'Concatenated BLOs\n')
JVIs = concat_load_metrics('JVI')
print(f'Concatenated JVIs\n')

### Loop through epochs
for start_yr, end_yr in epochs:

    ### Figure
    print(f'Historical Period: {start_yr}–{end_yr}')

    for letters in [True, False]:

        ### Time axis specifications
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
        month_positions = [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349, 349 + 30]
        month_positions = [pos - 14 for pos in month_positions]

        ### Sectors
        sector_options = ['NH (180°W–180°E)', 'AM (140°W–50°W)', 'AT (50°W–40°E)', 'AS (40°E–130°E)', 'PA (130°E–140°W)']
        colors = {'NH (180°W–180°E)': 'k', 'AM (140°W–50°W)': 'navy', 'AT (50°W–40°E)': 'forestgreen', 'AS (40°E–130°E)': 'firebrick', 'PA (130°E–140°W)': 'darkorange'}

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
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        plt.subplots_adjust(hspace=.25, wspace=.25)

        ### Title
        fig.suptitle(r'\textbf{Daily Diagnostics by Sector}', fontsize=26, fontweight='bold', y=.985)

        ### Letters and annotation
        letters_text = ['(a)', '(b)', '(c)', '(d)']
        fig.text(.02, .98, f'{start_yr}–{end_yr}', fontsize=20, fontweight='bold', color='dimgray', ha='left', va='top')

        ### Time specification
        ZON_data = ZONs.sel(time=slice(f'{start_yr}-01-01', f'{end_yr}-12-31'))
        SIN_data = SINs.sel(time=slice(f'{start_yr}-01-01', f'{end_yr}-12-31'))
        BLO_data = BLOs.sel(time=slice(f'{start_yr}-01-01', f'{end_yr}-12-31'))
        JVI_data = JVIs.sel(time=slice(f'{start_yr}-01-01', f'{end_yr}-12-31'))

        ### Calculate climatology
        ZON_climo, SIN_climo = calc_climo(ZON_data, 'ZON'), calc_climo(SIN_data, 'SIN')
        BLO_climo = calc_climo(BLO_data, 'BLO')

        ### JVI climatology (remove leap day to avoid discontinuities)
        JVIs_daily = JVI_data.sortby('time').where(~((JVI_data.time.dt.month == 2) & (JVI_data.time.dt.day == 29)), drop=True)
        JVI_climo = JVIs_daily.assign_coords(month_day=('time', JVIs_daily.time.dt.strftime('%m-%d').data)).groupby('month_day').mean(dim='time')
        JVI_climo = JVI_climo[[f"JVI_{sector.split(' ')[0]}" for sector in sector_options]]

        ##############################################
        ### Top-left panel : (a) Zonal Index (ZON) ###
        ##############################################
        ZON_title, ZON_y_label = r'\textbf{Zonal Index (ZON)}', r'\textbf{Geopotential Height Difference [gpm]}'
        ZON_ticks = np.arange(200, 650, 50)
        ZON_tick_labels = [f'{int(i)}' for i in ZON_ticks]
        ZON_limits, ZON_hlines = [200, 600], [(100, 'solid'), (50, 'dotted')]
        plot_panel(ax=axes[0, 0], climo=ZON_climo, index_prefix='ZON', title=ZON_title, y_ticks=ZON_ticks, y_tick_labels=ZON_tick_labels, y_lim=ZON_limits, hlines=ZON_hlines, y_label=ZON_y_label)

        ###################################################
        ### Top-right panel : (b) Sinuosity Index (SIN) ###
        ###################################################
        SIN_title, SIN_ticks = r'\textbf{Sinuosity Index (SIN)}', np.arange(0, 5, .1)
        SIN_tick_labels = [f'{i:.1f}' for i in SIN_ticks]
        SIN_limits, SIN_hlines = [1.2, 2.1], [(2, 'solid', 'single'), (.1, 'dotted')]
        plot_panel(ax=axes[0, 1], climo=SIN_climo, index_prefix='SIN', title=SIN_title, y_ticks=SIN_ticks, y_tick_labels=SIN_tick_labels, y_lim=SIN_limits, hlines=SIN_hlines)

        ####################################################
        ### Bottom-left panel : (c) Blocking Index (BLO) ###
        ####################################################
        BLO_title, BLO_ticks = r'\textbf{Blocking Index (BLO)}', np.arange(0, 1.1, .1)
        BLO_tick_labels = ['0' if i == 0 else (f'.{int(i*10)}' if i < 1 else '1') for i in BLO_ticks]
        BLO_limits, BLO_hlines, BLO_x_label, BLO_y_label = [.1, 1], [(1, 'solid', 'single'), (.1, 'dotted')], r'\textbf{Month}', r'\textbf{Probability}'
        plot_panel(ax=axes[1, 0], climo=BLO_climo, index_prefix='BLO', title=BLO_title, y_ticks=BLO_ticks, y_tick_labels=BLO_tick_labels,
                   y_lim=BLO_limits, hlines=BLO_hlines, x_label=BLO_x_label, y_label=BLO_y_label)

        ############################################################
        ### Bottom-right panel : (d) Jet Variability Index (JVI) ###
        ############################################################
        JVI_title, JVI_ticks = r'\textbf{Jet Variability Index (JVI)}', np.arange(0, 55, 5)
        JVI_tick_labels = [f'{int(i)}' for i in JVI_ticks]
        JVI_limits, JVI_hlines, JVI_x_label, JVI_y_label = [20, 35], [(5, 'solid'), (1, 'dotted')], r'\textbf{Month}', r'\textbf{JVI [° Latitude]}'
        plot_panel(ax=axes[1, 1], climo=JVI_climo, index_prefix='JVI', title=JVI_title, y_ticks=JVI_ticks, y_tick_labels=JVI_tick_labels,
                   y_lim=JVI_limits, hlines=JVI_hlines, x_label=JVI_x_label, y_label=JVI_y_label)

        ### Legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.995, .995), fontsize=8, handlelength=1, frameon=False, shadow=False, borderaxespad=.0, ncol=1, labelcolor='k')

        ### Save figure
        plt.tight_layout()
        if letters:
            for i, ax in enumerate(axes.flat):
                ax.text(-.08, 1.05, rf'\textbf{{{letters_text[i]}}}', fontweight='bold', ha='left', va='bottom', color='k', fontsize=17, transform=ax.transAxes)
        suffix = '_letters' if letters else ''
        fig_path = os.path.join(fig_dir, f'Reanalysis_Daily_Diagnostics_{start_yr}–{end_yr}_Climo_Sectors{suffix}.png')
        plt.savefig(fig_path, dpi=1000)
        plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')