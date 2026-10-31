"""
Plot Arctic monthly Sea Ice Volume (SIV) from the Pan-Arctic Ice Modeling and Assimilation System (PIOMAS) reanalysis

Notes
-----
    Data : https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib.colors import LinearSegmentedColormap
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
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/ICEFRAC/SIV/')

### Years
year_crop = 1980
year_min, year_max = 1979, 2025

### Time specifications
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan']
month_positions = [1, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365]

print(f'\n---- Arctic Sea Ice Volume ----\n')

### Load data
siv = xr.open_dataset(os.path.join(data_dir, f'PIOMAS_Arctic_{year_min}-{year_max}_SIV_Monthly.nc'))['SIV'].sortby('time')

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', labelcolor='dimgray', facecolor='w')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure
fig, ax = plt.subplots(figsize=(10, 6))

### Adjust axes
adjust_spines(ax, ['left', 'bottom'])
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

### Ticks
yticks = np.arange(0, 41, 5)
ax.set_xticks(month_positions, month_labels, fontsize=12, color='k', rotation=0)
ax.set_yticks(yticks, map(str, yticks), color='k')

### Domain
ax.set_xlim(1, 365); ax.set_ylim([0, 35])

### Horizontal lines
for step, style, lw in [(5, 'solid', .5), (1, 'dotted', .5)]:
    for y in np.arange(0, 41, step):
        plt.axhline(y, color='gray', linestyle=style, linewidth=lw)

### Plot
colors = ['#ffeda0', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000', '#000000']
years_sorted = np.unique(siv['time.year'].values)
num_years = len(years_sorted)
blue_cmap = LinearSegmentedColormap.from_list('blue_cmap', colors, num_years)
for idx, year in enumerate(years_sorted):

    ### Start with months Jan–Dec
    group = siv.sel(time=str(year))
    full_year = group.values.copy()
    months = month_positions[:-1]

    ### Fetch previous December if it exists
    if idx > 0:
        prev_dec = siv.sel(time=f'{years_sorted[idx - 1]}-12').item()
        full_year = np.concatenate([[prev_dec], full_year])
        months = [month_positions[0] - 31] + months

    ### Fetch next January if it exists
    if idx < len(years_sorted) - 1:
        next_jan = siv.sel(time=f'{years_sorted[idx + 1]}-01').item()
        full_year = np.concatenate([full_year, [next_jan]])
        months = months + [month_positions[-1] + 31]

    plt.plot(months, full_year, color=blue_cmap(idx / (num_years - 1)))

### Title and labels
plt.title(r'\textbf{ARCTIC SEA ICE VOLUME}', color='k', fontsize=30)
plt.xlabel(r'\textbf{Month}', fontsize=13, color='k', labelpad=10)
plt.ylabel(r'\textbf{Sea Ice Volume [10\textsuperscript{{3}} km\textsuperscript{{3}}]}', fontsize=13, color='k', labelpad=10)

### Legend
decades = [year for year in years_sorted if year % 10 == 0]
l_handles = [plt.Line2D([0], [0], color=blue_cmap(i / (len(decades) - 1)), lw=2) for i, _ in enumerate(decades)]
l = plt.legend(handles=l_handles, labels=[f'{year}s' for year in decades], shadow=False, fontsize=4, loc='upper left',
               bbox_to_anchor=(.62, .881), fancybox=True, ncol=1, frameon=False, prop={'weight': 'bold'}, labelspacing=0)

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_PIOMAS_SIV_{year_crop}–{year_max}.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')