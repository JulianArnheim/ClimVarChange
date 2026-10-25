"""
Plot Jet Variability Index (JVI) climatology from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
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
year_min, year_max = 1940, 2025

### Epochs
epochs = [(year_min, year_max), (1981, 2010)]

print(f'\n---- Jet Variability Index (JVI) Climatology ----\n')

### Load JVI data
JVI_datasets = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_JVI_Sectors_{start_year}-{end_year}_Daily.nc')
    if os.path.exists(filename):
        JVI_datasets.append(xr.open_dataset(filename))
        print(f'Loaded: {start_year}–{end_year}')
JVIs = xr.concat(JVI_datasets, dim='time')
print(f'Concatenated JVIs')

### Loop through epochs
for start_yr, end_yr in epochs:

    ### Time specification
    JVIs_time = JVIs.sel(time=slice(f'{start_yr}-01-01', f'{end_yr}-12-31'))

    ### Time axis specifications
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
    month_positions = [pos - 14 for pos in [15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349, 349 + 30]]

    ### Sectors
    sector_options = ['NH (180°W–180°E)', 'AM (140°W–50°W)', 'AT (50°W–40°E)', 'AS (40°E–130°E)', 'PA (130°E–140°W)']

    ### Colors for each sector
    colors = {'NH (180°W–180°E)': 'k',
              'AM (140°W–50°W)': 'navy',
              'AT (50°W–40°E)': 'forestgreen',
              'AS (40°E–130°E)': 'firebrick',
              'PA (130°E–140°W)': 'darkorange'}

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
    for key in ['top', 'right']:
        ax.spines[key].set_color('none')
    for key in ['bottom', 'left']:
        ax.spines[key].set_linewidth(2)
    ax.tick_params(axis='both', length=5.5, width=2, which='major', colors='dimgray')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('k')

    ### Horizontal lines
    for step, style in [(5, 'solid'), (1, 'dotted')]:
        for y in np.arange(0, 50, step):
            ax.axhline(y, color='gray', linestyle=style, linewidth=.5, zorder=1)

    ### Process JVI climatology
    JVIs_daily = JVIs_time.sortby('time')
    JVIs_daily = JVIs_daily.where(~((JVIs_daily.time.dt.month == 2) & (JVIs_daily.time.dt.day == 29)), drop=True)
    JVIs_daily = JVIs_daily.assign_coords(month_day=('time', JVIs_daily.time.dt.strftime('%m-%d').data))
    climo_JVIs_daily_year = JVIs_daily.groupby('month_day').mean(dim='time')

    ### Plot
    days = np.arange(1, 366)
    for sector in sector_options:
        timeseries = climo_JVIs_daily_year[f'JVI_{sector.split(" ")[0]}']
        if sector.startswith('NH'):
            plt.plot(days, timeseries, color=colors[sector], label=sector, linewidth=3, zorder=3)
        else:
            plt.plot(days, timeseries, color=colors[sector], label=sector, linewidth=3, zorder=2)

    ### Title
    plt.title(r'\textbf{Jet Variability Index (JVI)}', color='k', fontsize=25, y=1.01)

    ### Ticks
    latitude_ticks = np.arange(0, 55, 5)
    plt.xticks(month_positions, month_labels, fontsize=12, color='k', rotation=0)
    plt.yticks(latitude_ticks, latitude_ticks, fontsize=12, color='k')

    ### Domain
    plt.axis([1, 365, 20, 35])

    ### Labels
    plt.xlabel(r'\textbf{Month}', fontsize=16, color='k', labelpad=10)
    plt.ylabel(r'\textbf{Latitude [°N]}', fontsize=16, color='k', labelpad=10)

    ### Annotation
    plt.text(.5, .95, f'{start_yr}–{end_yr} Climatology', color='k', fontsize=14, ha='center', transform=plt.gca().transAxes)

    ### Legend
    l = plt.legend(shadow=False, fontsize=13, loc='upper center', bbox_to_anchor=(.22, .415), frameon=False, labelspacing=.75)
    for text in l.get_texts():
        text.set_color('k')

    ### Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_JVI_{start_yr}–{end_yr}_Climo_Sectors.png'), dpi=1000)
    plt.close(fig)

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')