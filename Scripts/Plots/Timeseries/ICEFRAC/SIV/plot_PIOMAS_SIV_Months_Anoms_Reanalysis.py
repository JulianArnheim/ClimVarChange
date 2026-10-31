"""
Plot Arctic monthly Sea Ice Volume (SIV) anomalies from the Pan-Arctic Ice Modeling and Assimilation System (PIOMAS) reanalysis

Notes
-----
    Data : https://psc.apl.uw.edu/research/projects/arctic-sea-ice-volume-anomaly/data/
    Author : Julian B Arnheim
"""

### Import modules
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

def custom_formatter(x, _):
    if abs(x) < .01:
        return '0'
    return f'{x:+.0f}'

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/ICEFRAC/SIV/')

### Years and epoch
year_crop = 1980
year_min, year_max = 1979, 2025
epoch = (1981, 2010)

### Time specifications
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
colors = ['#B0C4DE', '#9ACD32', '#FFC0CB', '#D2691E', '#00FFFF', '#FFA500',
          '#FF69B4', '#32CD32', '#FFD700', '#800080', '#FF0000', '#7B68EE']

print(f'\n---- Arctic Sea Ice Volume ----\n')

### Load data
siv = xr.open_dataset(os.path.join(data_dir, f'PIOMAS_Arctic_{year_min}-{year_max}_SIV_Monthly.nc'))['SIV'].sortby('time')

### Annual mean sea ice volume anomalies
baseline_volume = siv.sel(time=slice(f'{epoch[0]}-01-01', f'{epoch[1]}-12-31')).mean(dim='time')
annual_mean_anom = siv.groupby('time.year').mean(dim='time') - baseline_volume

### Monthly sea ice volume anomalies
SIV_by_month = {}
for month, group in siv.groupby('time.month'):
    monthly_refmean = group.sel(time=slice(f'{epoch[0]}-01-01', f'{epoch[1]}-12-31')).mean(dim='time')
    SIV_by_month[month] = group - monthly_refmean

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
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=12)

### Ticks and domain
x_ticks, y_ticks = np.arange(year_crop, 2026, 5), np.arange(-10, 5.1, 5)
plt.xticks(x_ticks, x_ticks, color='k')
plt.yticks(y_ticks, map(str, y_ticks), color='k')
plt.axis([year_crop, 2025, -10, 5])

### Horizontal lines
for step, style, lw in [(5, 'solid', .5), (1, 'dotted', .5)]:
    for y in np.arange(-20, 20, step):
        plt.axhline(y, color='gray', linestyle=style, linewidth=1 if y == 0 else lw)

### Plot
for month, anom in SIV_by_month.items():
    plt.plot(anom['time.year'], anom, color=colors[month - 1], label=months[month - 1])
plt.plot(annual_mean_anom['year'], annual_mean_anom.values, color='k', linewidth=3)

### Title
plt.title(r'\textbf{ARCTIC SEA ICE VOLUME}', color='k', fontsize=30)

### Labels
plt.xlabel(r'\textbf{Year}', fontsize=13, color='k', labelpad=10)
plt.ylabel(r'\textbf{Sea Ice Volume Anomaly [10\textsuperscript{{3}} km\textsuperscript{{3}}]}', fontsize=13, color='k', labelpad=10)

### Annotation
plt.text(2025, .1, rf'\textbf{{BASELINE: {epoch[0]}–{epoch[1]}}}', fontsize=11, ha='right', color='k')

### Legend
l = plt.legend(shadow=False, fontsize=11, loc='upper center', bbox_to_anchor=(.5, 1.026), fancybox=True, ncol=6, frameon=False, labelspacing=0)
for text in l.get_texts():
    text.set_color('k')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(custom_formatter))

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_PIOMAS_SIV_Anoms_Months_{year_crop}–{year_max}.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')