"""
Plot the monthly Oceanic Niño Index (ONI) and Relative Oceanic Niño Index (RONI), three-month running means of
Niño 3.4 (5°N–5°S, 120°–170°W) SST anomalies (ONI from ERSSTv6, RONI from ERSSTv5), with El Niño and La Niña strength categories

Notes
-----
    Data : https://psl.noaa.gov/data/correlation/oni.data
    Data : https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt
    Author : Julian B Arnheim
"""

### Import modules
import matplotlib.pyplot as plt, matplotlib.patheffects as pe, matplotlib.transforms as transforms
import numpy as np
import os
import pandas as pd
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

def plot_index(ax, years, index, linestyle, lw_neutral, lw_event):
    ax.plot(years, index, color='k', linestyle=linestyle, linewidth=lw_neutral, zorder=3)
    ax.plot(years, np.where(index >= .5, index, np.nan), color=nino_color, linestyle=linestyle, linewidth=lw_event, zorder=4)
    ax.plot(years, np.where(index <= -.5, index, np.nan), color=nina_color, linestyle=linestyle, linewidth=lw_event, zorder=4)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/ENSO/')

### Years
year_min, year_max = 1950, 2026

### Strength categories [°C]
categories = [('Weak', .5, 1), ('Moderate', 1, 1.5), ('Strong', 1.5, 2), ('Very Strong', 2, 3)]

### Colors
nino_color, nina_color = 'crimson', 'deepskyblue'

print(f'\n---- Oceanic Niño Index (ONI) and Relative Oceanic Niño Index (RONI) ----\n')

### Seasons
seasons = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']

### Load data
filename = os.path.join(data_dir, f'NOAA_Nino34_{year_min}-{year_max}_ONI_RONI_Monthly.nc')
ds = xr.open_dataset(filename)
times = pd.to_datetime(ds['time'].values)
years = times.year + (times.month - .5) / 12
ONI, RONI = ds['ONI'].values, ds['RONI'].values

### Latest season of each index
last_oni, last_roni = np.where(np.isfinite(ONI))[0][-1], np.where(np.isfinite(RONI))[0][-1]
season_oni, season_roni = (f'{seasons[times[i].month - 1]}' for i in (last_oni, last_roni))
latest = f'{ONI[last_oni]:+.1f}'
latest += f' ({RONI[last_oni]:+.1f})' if np.isfinite(RONI[last_oni]) else ''

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', labelcolor='dimgray', facecolor='w')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure
fig, ax = plt.subplots(figsize=(14, 5))

### Adjust axes
adjust_spines(ax, ['left', 'bottom'])
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelsize=11)

### Ticks
plt.xticks(np.arange(1950, 2031, 5), np.arange(1950, 2031, 5), color='k')
ax.set_yticks(np.arange(-3, 3.1, 1))
ax.set_yticklabels([f'{y:.0f}' for y in np.arange(-3, 3.1, 1)], color='k', fontsize=11)
ax.set_yticks(np.arange(-2.5, 3, .5), minor=True)
ax.tick_params(axis='y', which='minor', length=3.5, width=1.5, color='dimgray')

### Domain
ax.axis([year_min, 2030, -3, 3])

### Strength category shading and labels
for i, (name, low, high) in enumerate(categories):
    for sign, color in [(1, nino_color), (-1, nina_color)]:
        horizontal_shading, plot_shading = .05 + .04 * i, .4 + .2 * i
        ax.axhspan(sign * low, sign * high, color=color, alpha=horizontal_shading, linewidth=0, zorder=0)
        color_transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        height = sign * (low + high) / 2
        plt.text(1.005, height, rf'\textbf{{{name}}}', transform=color_transform, color=color, alpha=plot_shading, fontsize=10, va='center', ha='left')

### Threshold lines
for level in [.5, 1, 1.5, 2, 3]:
    ax.axhline(level, color='dimgray', linestyle='--', linewidth=.6, alpha=.8, zorder=2)
    ax.axhline(-level, color='dimgray', linestyle='--', linewidth=.6, alpha=.8, zorder=2)

### Zero line
ax.axhline(0, color='dimgray', linewidth=1, zorder=1)

### Event names
plt.text(.5, .955, r'\textbf{El Ni\~no}', transform=ax.transAxes, color=nino_color, fontsize=20, ha='center', va='top', zorder=5)
plt.text(.5, .045, r'\textbf{La Ni\~na}', transform=ax.transAxes, color=nina_color, fontsize=20, ha='center', va='bottom', zorder=5)

### Plot
plot_index(ax, years, RONI, (0, (1, .8)), 1.25, 2)
plot_index(ax, years, ONI, '-', 1.25, 2)
plt.scatter(years[last_oni], ONI[last_oni], s=40, color='gold', edgecolor='k', linewidth=.8, zorder=6)
ax.annotate(rf'\textbf{{{season_oni}: {latest} °C}}', xy=(1, 1.042), xycoords='axes fraction', ha='right', va='top',
            color='gold', fontsize=12, path_effects=[pe.withStroke(linewidth=1.5, foreground='k')], zorder=8)

### Label significant ENSO events
event_labels = {r"\textbf{'55–'56}": (1958.2, -1.75), r"\textbf{'72–'73}": (1972.8, 2.25), r"\textbf{'73–'74}": (1973.9, -2.28), r"\textbf{'82–'83}": (1980.7, 2.5),
                r"\textbf{'88–'89}": (1991.3, -1.83), r"\textbf{'97–'98}": (2000.2, 2.45), r"\textbf{'07–'08}": (2005.6, -1.8),  r"\textbf{'09–'10}": (2009.8, 1.7),
                r"\textbf{'10–'11}": (2013.5, -1.7),  r"\textbf{'15–'16}": (2018.3, 2.55), r"\textbf{'23–'24}": (2021.8, 2.15)}
for label, (x, y) in event_labels.items():
    ax.text(x, y, label, color='#E6B800', fontsize=11, ha='center', va='center', zorder=7)

### Legend
l_handles = [plt.Line2D([0], [0], color='k', linestyle='-', lw=2.5), plt.Line2D([0], [0], color='k', linestyle=(0, (1, .8)), lw=1.5)]
plt.legend(l_handles, ['ONI', 'RONI'], shadow=False, fontsize=11, loc='upper left', bbox_to_anchor=(.01, 1.01), fancybox=True, ncol=1, frameon=False, handlelength=2)

### Title
plt.title(r'\textbf{EL NI\~NO SOUTHERN OSCILLATION: OCEANIC NI\~NO INDEX}', color='k', y=1.06, fontsize=28)

### Label
plt.ylabel(r'\textbf{Index [°C]}', fontsize=15, color='k', labelpad=8)

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_ENSO_ONI_RONI_{year_min}–{year_max}.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')