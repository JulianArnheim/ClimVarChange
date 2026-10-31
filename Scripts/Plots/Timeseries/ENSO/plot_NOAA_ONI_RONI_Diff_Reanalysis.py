"""
Plot the monthly Oceanic Niño Index (ONI) and Relative Oceanic Niño Index (RONI), three-month running means of
Niño 3.4 (5°N–5°S, 120°–170°W) SST anomalies (ONI from ERSSTv6, RONI from ERSSTv5), with El Niño and La Niña strength categories, and RONI minus ONI

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

### Strength categories [°C] and colors
categories, (nino_color, nina_color) = [('Weak', .5, 1), ('Moderate', 1, 1.5), ('Strong', 1.5, 2), ('Very Strong', 2, 3)], ('crimson', 'deepskyblue')

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
last_both = min(last_oni, last_roni)

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', labelcolor='dimgray', facecolor='w')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure
fig, (ax, ax_diff) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [4, 1.25], 'hspace': .28})

######################################
### Top panel : ONI and RONI index ###
######################################

### Adjust axes
adjust_spines(ax, ['left', 'bottom'])
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelcolor='k', labelsize=11)

### Ticks
ax.set_yticks(np.arange(-3, 3.1, 1), [f'{y:.0f}' for y in np.arange(-3, 3.1, 1)], color='k', fontsize=11)
ax.set_yticks(np.arange(-2.5, 3, .5), minor=True)
ax.tick_params(axis='y', which='minor', length=3.5, width=1.5, color='dimgray')

### Domain
ax.axis([year_min, 2030, -3, 3])

### Strength category shading and labels
for i, (name, low, high) in enumerate(categories):
    for sign, color in [(1, nino_color), (-1, nina_color)]:
        ax.axhspan(sign * low, sign * high, color=color, alpha=.05 + .04 * i, linewidth=0, zorder=0)
        transform = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.005, sign * (low + high) / 2, rf'\textbf{{{name}}}', transform=transform, color=color, alpha=.4 + .2 * i, fontsize=10, va='center', ha='left')

### Threshold and zero lines
for level in [.5, 1, 1.5, 2, 3]:
    ax.axhline(level, color='dimgray', linestyle='--', linewidth=.6, alpha=.8, zorder=2)
    ax.axhline(-level, color='dimgray', linestyle='--', linewidth=.6, alpha=.8, zorder=2)
ax.axhline(0, color='dimgray', linewidth=1, zorder=1)

### Event names
ax.text(.5, .955, r'\textbf{El Ni\~no}', transform=ax.transAxes, color=nino_color, fontsize=20, ha='center', va='top', zorder=5)
ax.text(.5, .036, r'\textbf{La Ni\~na}', transform=ax.transAxes, color=nina_color, fontsize=20, ha='center', va='bottom', zorder=5)

### Plot
plot_index(ax, years, ONI, '-', 1.25, 2)
plot_index(ax, years, RONI, (0, (1, .8)), 1.25, 2)
ax.scatter(years[last_oni], ONI[last_oni], s=40, color='gold', edgecolor='k', linewidth=.8, zorder=6)
ax.annotate(rf'\textbf{{{season_oni}: {latest} °C}}', xy=(1, 1.045), xycoords='axes fraction', ha='right', va='top',
            color='gold', fontsize=12, path_effects=[pe.withStroke(linewidth=1.5, foreground='k')], zorder=8)

### Label significant ENSO events
event_labels = {r"\textbf{'55–'56}": (1958.4, -1.75), r"\textbf{'72–'73}": (1972.8, 2.2),  r"\textbf{'73–'74}": (1973.9, -2.28), r"\textbf{'82–'83}": (1980.7, 2.5),
                r"\textbf{'88–'89}": (1991.5, -1.83), r"\textbf{'97–'98}": (2000.5, 2.45), r"\textbf{'07–'08}": (2005.6, -1.8),  r"\textbf{'09–'10}": (2009.8, 1.7),
                r"\textbf{'10–'11}": (2013.5, -1.7),  r"\textbf{'15–'16}": (2018.4, 2.62), r"\textbf{'23–'24}": (2021.8, 2.15)}
for label, (x, y) in event_labels.items():
    ax.text(x, y, label, color='#E6B800', fontsize=10, ha='center', va='center', zorder=7)

### Legend
l_handles = [plt.Line2D([0], [0], color='k', linestyle='-', lw=2.5), plt.Line2D([0], [0], color='k', linestyle=(0, (1, .8)), lw=1.5)]
ax.legend(l_handles, ['ONI', 'RONI'], shadow=False, fontsize=11, loc='upper left', bbox_to_anchor=(.01, 1.01), fancybox=True, ncol=1, frameon=False, handlelength=2)

### Title
ax.set_title(r'\textbf{EL NI\~NO SOUTHERN OSCILLATION: OCEANIC NI\~NO INDEX}', color='k', y=1.07, fontsize=29)

### Label
ax.set_ylabel(r'\textbf{Index [°C]}', fontsize=15, color='k', labelpad=8)

###########################################
### Bottom panel : RONI minus ONI index ###
###########################################

### Adjust axes
adjust_spines(ax_diff, ['left', 'bottom'])
ax_diff.spines['bottom'].set_linewidth(2)
ax_diff.spines['left'].set_linewidth(2)

### Plot RONI minus ONI
ax_diff.plot(years, RONI - ONI, color='k', linewidth=1.25, zorder=3)
ax_diff.scatter(years[last_both], RONI[last_both] - ONI[last_both], s=40, color='gold', edgecolor='k', linewidth=.8, zorder=6)

### Horizontal lines
for level in np.arange(-.9, .91, .3):
    ax_diff.axhline(level, color='dimgray', linestyle='--', linewidth=.6, alpha=.8, zorder=1)
ax_diff.axhline(0, color='dimgray', linewidth=1, zorder=2)

### Ticks
ax_diff.set_xticks(np.arange(1950, 2031, 5), np.arange(1950, 2031, 5), color='k')
ax_diff.set_yticks([-.9, 0, .9], ['-.9', '0', '+.9'], color='k', fontsize=11); ax_diff.set_yticks([-.6, -.3, .3, .6], minor=True)
ax_diff.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelcolor='k', labelsize=11)
ax_diff.tick_params(axis='y', which='minor', length=3.5, width=1.5, color='dimgray')

### Domain
ax_diff.set_ylim([-.9, .9])

### Label
ax_diff.set_ylabel(r'\textbf{RONI $-$ ONI [°C]}', fontsize=12, color='k', labelpad=8)

### Save figure
plt.savefig(os.path.join(fig_dir, f'Reanalysis_ENSO_ONI_RONI_Diff_{year_min}–{year_max}.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')