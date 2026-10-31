"""
Plot scatterplot between monthly teleconnection patterns from NOAA CPC indices

Notes
-----
    Data : https://www.cpc.ncep.noaa.gov/data/teledoc/telecontents.shtml
    Author : Julian B Arnheim
"""

### Import modules
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt, matplotlib.ticker as ticker
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

def annotate_extreme(idx, x_vals, y_vals, offset, fontsize=8):
    month_str = str(common_times[idx])[:7]
    point = (x_vals[idx], y_vals[idx])
    plt.annotate(month_str, point, xytext=offset, textcoords='offset points', arrowprops=dict(facecolor='k', arrowstyle='->'), fontsize=fontsize)

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Scatter/AT/')

### Years
year_min, year_max = 1950, 2025

print(f'\n---- North Atlantic Oscillation (NAO) & East Atlantic (EA) Pattern ----\n')

### Load data
filename = os.path.join(data_dir, f'CPC_NAO_EA_{year_min}-{year_max}_Monthly.nc')
teleconnections = xr.open_dataset(filename)

### Extract indices over complete DJF winters (Dec of year_min to Feb of year_max) and common valid months
NAO, EA = xr.align(*[teleconnections[v].sel(time=slice(f'{year_min}-12', f'{year_max}-02')) for v in ('NAO', 'EA')], join='inner')
valid = NAO.time.dt.month.isin([12, 1, 2]) & np.isfinite(NAO) & np.isfinite(EA)
NAO, EA = NAO.where(valid, drop=True), EA.where(valid, drop=True)
common_times, x, y = NAO.time.values, NAO.values, EA.values

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure
fig, ax = plt.subplots(figsize=(5.5, 4.5))

### Adjust axes
adjust_spines(ax, ['left', 'bottom'])
for key in ['top', 'right']:
    ax.spines[key].set_color('none')
for key in ['bottom', 'left']:
    ax.spines[key].set_linewidth(2)
ax.tick_params(axis='both', length=5.5, width=2, which='major', colors='dimgray')
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_color('k')
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:g}'))

### Grid lines
for i in np.arange(-5, 6):
    plt.axvline(i, color='dimgray', linestyle='dotted' if i % 5 else 'solid', linewidth=.5, zorder=1)
for j in np.arange(-5, 6):
    plt.axhline(j, color='dimgray', linestyle='dotted' if j % 5 else 'solid', linewidth=.5, zorder=1)

### Plot points colored by decade of the winter (January) year
point_times = common_times.astype('datetime64[M]')
point_months = point_times.astype(int) % 12 + 1
point_years = point_times.astype('datetime64[Y]').astype(int) + 1970 + (point_months == 12)  # December belongs to the following winter
point_decades = point_years // 10 * 10
decades = np.unique(point_decades)
colors = list(plt.cm.turbo(np.linspace(.1, .9, len(decades) - 1))) + ['k']  # incomplete final decade in black
for decade, color in zip(decades, colors):
    plt.scatter(x[point_decades == decade], y[point_decades == decade], color=color, s=6, zorder=3)

### Legend patches by decade
legend_patches = [mpatches.Patch(color=color, label=rf'\textbf{{{decade}s}}') for decade, color in zip(decades, colors)]

### Ticks and domain
plt.xticks(np.arange(-4, 5, 1), [f'{tick:.0f}' for tick in np.arange(-4, 5, 1)], color='k')
plt.yticks(np.arange(-4, 5, 1), [f'{tick:.0f}' for tick in np.arange(-4, 5, 1)], color='k')
ax.set(xlim=(-4, 4), ylim=(-4, 4))

### Title, annotations, and labels
plt.title(r'\textbf{North Atlantic Teleconnection Patterns}', color='k', fontsize=16, y=1.06)
plt.text(.5, 1.02, f'DJF {year_min + 1}–{year_max}', color='k', fontsize=12, ha='center', transform=plt.gca().transAxes)
plt.xlabel(r'\textbf{NAO}', fontsize=13, color='k', labelpad=8)
plt.ylabel(r'\textbf{EA}', fontsize=13, color='k', labelpad=8)

### Legend and annotations
plt.legend(handles=legend_patches, loc='lower right', bbox_to_anchor=(1.011, .025), prop={'size': 6}, frameon=False, labelspacing=.65)
max_NAO_idx, min_NAO_idx, max_EA_idx, min_EA_idx = x.argmax(), x.argmin(), y.argmax(), y.argmin()
annotate_extreme(max_NAO_idx, x, y, offset=(-3, 13))
annotate_extreme(min_NAO_idx, x, y, offset=(-30, -15))
annotate_extreme(max_EA_idx, x, y, offset=(8, 13))
annotate_extreme(min_EA_idx, x, y, offset=(-50, -10))

### Save figure
ax.set_position([.14, .15, .78, .72])
plt.savefig(os.path.join(fig_dir, f'Reanalysis_NAO_EA_DJF_{year_min + 1}–{year_max}.png'), dpi=1000)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')