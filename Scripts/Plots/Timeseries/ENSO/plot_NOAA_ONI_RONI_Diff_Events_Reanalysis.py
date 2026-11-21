"""
Plot the monthly Oceanic Niño Index (ONI) and Relative Oceanic Niño Index (RONI), three-month running means of
Niño 3.4 (5°N–5°S, 120°–170°W) SST anomalies (ONI from ERSSTv6, RONI from ERSSTv5), with El Niño and La Niña strength categories, RONI minus ONI,
and the JJA of year 1 to JJA of year 2 evolution of major El Niño events in each index

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

def event_label(year):
    return rf"\textbf{{'{year % 100:02d}–'{(year + 1) % 100:02d}}}"

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

def major_events(series):
    return [y for y in range(year_min, year_now) if series.loc[f'{y}-10-01':f'{y + 1}-03-01'].max() >= major_threshold]  # peak from SON to FMA

def plot_index(ax, years, index, linestyle, lw_neutral, lw_event):
    x, y = np.asarray(years), np.asarray(index)
    for c in (-.5, .5):
        i = np.where((index[:-1] - c) * (index[1:] - c) < 0)[0]  # months just before each crossing of the threshold
        x, y = np.append(x, x[i] + (c - y[i]) / (y[i + 1] - y[i]) * (x[i + 1] - x[i])), np.append(y, np.full(len(i), c))
    x, y = x[np.argsort(x)], y[np.argsort(x)]
    for color, linewidth, zorder, mask in [('k', lw_neutral, 3, np.abs(y) <= .5), (nino_color, lw_event, 4, y >= .5), (nina_color, lw_event, 4, y <= -.5)]:
        ax.plot(x, np.where(mask, y, np.nan), color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder)

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

### Major El Niño threshold [°C] and event colors
major_threshold, current_color = 1.5, 'magenta'
event_colors = ['#4E79A7', '#F28E2B', '#59A14F', '#76B7B2', '#B6992D', '#9C755F', '#A0CBE8', '#8CD17D', '#79706E', '#FFBE7D', '#499894', '#BAB0AC']

print(f'\n---- Oceanic Niño Index (ONI) and Relative Oceanic Niño Index (RONI) ----\n')

### Seasons
seasons = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ', 'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']

### Load data
ds = xr.open_dataset(os.path.join(data_dir, f'NOAA_Nino34_{year_min}-{year_max}_ONI_RONI_Monthly.nc'))
times = pd.to_datetime(ds['time'].values)
years, ONI, RONI = times.year + (times.month - .5) / 12, ds['ONI'].values, ds['RONI'].values

### Latest season of each index
last_oni, last_roni = np.where(np.isfinite(ONI))[0][-1], np.where(np.isfinite(RONI))[0][-1]
season_oni, season_roni = (f'{seasons[times[i].month - 1]}' for i in (last_oni, last_roni))
latest = f'{ONI[last_oni]:+.1f}'
latest += f' ({RONI[last_oni]:+.1f})' if np.isfinite(RONI[last_oni]) else ''
last_both = min(last_oni, last_roni)

### Current event, which starts in JJA of the latest year once that season is available
year_now = times[last_oni].year if times[last_oni].month >= 7 else times[last_oni].year - 1

### Major El Niño events selected by ONI and shown in both panels, with one color per event
index_series = {'ONI': pd.Series(ONI, index=times), 'RONI': pd.Series(RONI, index=times)}
colors = {y: event_colors[i % len(event_colors)] for i, y in enumerate(major_events(index_series['ONI']))}
print(f'Major El Niño events (ONI): {", ".join(f"{y}–{y + 1}" for y in colors)}')
print(f'Number of events: {len(colors)}\n')

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray', labelcolor='dimgray', facecolor='w')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure
fig = plt.figure(figsize=(12, 10))
grid = fig.add_gridspec(5, 2, height_ratios=[52, 9.5, 15, 16.5, 33], hspace=0, wspace=.14, bottom=.07, top=.905)  # panels 52/15/33, rows 2 and 4 are gaps
ax = fig.add_subplot(grid[0, :])
ax_diff = fig.add_subplot(grid[2, :], sharex=ax)
ax_oni, ax_roni = fig.add_subplot(grid[4, 0]), fig.add_subplot(grid[4, 1])

######################################
### Top panel : ONI and RONI index ###
######################################

### Adjust axes
adjust_spines(ax, ['left', 'bottom'])
ax.spines['bottom'].set_linewidth(2); ax.spines['left'].set_linewidth(2)
ax.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelcolor='k', labelsize=11)

### Ticks
ax.set_yticks(np.arange(-3, 3.1, 1), [f'{y:.0f}' for y in np.arange(-3, 3.1, 1)], color='k', fontsize=11); ax.set_yticks(np.arange(-2.5, 3, .5), minor=True)
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
    ax.axhline(level, color='dimgray', linestyle='--', linewidth=.6, alpha=.8, zorder=2); ax.axhline(-level, color='dimgray', linestyle='--', linewidth=.6, alpha=.8, zorder=2)
ax.axhline(0, color='dimgray', linewidth=1, zorder=1)

### Event names
ax.text(.5, .955, r'\textbf{El Ni\~no}', transform=ax.transAxes, color=nino_color, fontsize=20, ha='center', va='top', zorder=5)
ax.text(.5, .036, r'\textbf{La Ni\~na}', transform=ax.transAxes, color=nina_color, fontsize=20, ha='center', va='bottom', zorder=5)

### Plot
plot_index(ax, years, ONI, '-', 1.2, 1.5)
plot_index(ax, years, RONI, (0, (1, .8)), 1.2, 1.5)
ax.scatter(years[last_oni], ONI[last_oni], s=40, color='gold', edgecolor='k', linewidth=.8, zorder=6)
ax.annotate(rf'\textbf{{{season_oni}: {latest} °C}}', xy=(1, 1.055), xycoords='axes fraction', ha='right', va='top',
            color='gold', fontsize=12, path_effects=[pe.withStroke(linewidth=1.5, foreground='k')], zorder=8)

### Label significant ENSO events
event_labels = {r"\textbf{'55–'56}": (1958.4, -1.75), r"\textbf{'72–'73}": (1972.8, 2.2),  r"\textbf{'73–'74}": (1973.9, -2.28), r"\textbf{'82–'83}": (1980.7, 2.5),
                r"\textbf{'88–'89}": (1991.5, -1.83), r"\textbf{'97–'98}": (2000.5, 2.45), r"\textbf{'07–'08}": (2005.6, -1.8),  r"\textbf{'09–'10}": (2009.8, 1.7),
                r"\textbf{'10–'11}": (2013.5, -1.7),  r"\textbf{'15–'16}": (2018.4, 2.62), r"\textbf{'23–'24}": (2021.8, 2.15)}
for label, (x, y) in event_labels.items():
    ax.text(x, y, label, color='#E6B800', fontsize=10, ha='center', va='center', zorder=7)

### Legend
l_handles = [plt.Line2D([0], [0], color='k', linestyle='-', lw=2.5), plt.Line2D([0], [0], color='k', linestyle=(0, (1, .8)), lw=1.5)]
ax.legend(l_handles, ['ONI', 'RONI'], shadow=False, fontsize=11, loc='upper left', bbox_to_anchor=(.01, 1.01), fancybox=True, ncol=1, frameon=False, handlelength=2, labelspacing=.2)

### Title
ax.set_title(r'\textbf{EL NI\~NO SOUTHERN OSCILLATION: OCEANIC NI\~NO INDEX}', color='k', y=1.1, fontsize=29)

### Label
ax.set_ylabel(r'\textbf{Index [°C]}', fontsize=15, color='k', labelpad=8)

### Data source annotation
ax.text(0, 1.03, r'\textbf{DATA:} NOAA PSL (ONI, ERSSTv6), NOAA CPC (RONI, ERSSTv5). 3-month running means, centered.', transform=ax.transAxes, color='dimgray', fontsize=6, ha='left', va='top')

###########################################
### Middle panel : RONI minus ONI index ###
###########################################

### Adjust axes
adjust_spines(ax_diff, ['left', 'bottom'])
ax_diff.spines['bottom'].set_linewidth(2)
ax_diff.spines['left'].set_linewidth(2)

### Plot RONI minus ONI
ax_diff.plot(years, RONI - ONI, color='k', linewidth=1.25, zorder=3)
ax_diff.scatter(years[last_both], RONI[last_both] - ONI[last_both], s=40, color='gold', edgecolor='k', linewidth=.8, zorder=6)

### Horizontal lines
for level in np.arange(-.8, .81, .4):
    ax_diff.axhline(level, color='dimgray', linestyle='--', linewidth=.6, alpha=.8, zorder=1)
ax_diff.axhline(0, color='dimgray', linewidth=1, zorder=2)

### Ticks
ax_diff.set_xticks(np.arange(1950, 2031, 5), np.arange(1950, 2031, 5), color='k')
ax_diff.set_yticks([-.8, 0, .8], ['-.8', '0', '+.8'], color='k', fontsize=11); ax_diff.set_yticks([-.4, .4], minor=True)
ax_diff.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelcolor='k', labelsize=11); ax_diff.tick_params(axis='y', which='minor', length=3.5, width=1.5, color='dimgray')

### Domain
ax_diff.set_ylim([-.8, .8])

### Label
ax_diff.set_ylabel(r'\textbf{RONI $-$ ONI [°C]}', fontsize=12, color='k', labelpad=8)

####################################################################
### Bottom panels : major El Niño events, JJA to JJA (ONI, RONI) ###
####################################################################

for ax_event, name in [(ax_oni, 'ONI'), (ax_roni, 'RONI')]:
    series, last = index_series[name], (last_oni if name == 'ONI' else last_roni)

    ### Adjust axes
    adjust_spines(ax_event, ['left', 'bottom'])
    ax_event.spines['bottom'].set_linewidth(2)
    ax_event.spines['left'].set_linewidth(2)

    ### Strength category shading, threshold lines and zero line
    for i, (_, low, high) in enumerate(categories):
        ax_event.axhspan(low, high, color=nino_color, alpha=.05 + .04 * i, linewidth=0, zorder=0)
        ax_event.axhspan(-high, -low, color=nina_color, alpha=.05 + .04 * i, linewidth=0, zorder=0)
    for level in [-1.5, -1, -.5, .5, 1, 1.5, 2, 2.5, 3]:
        ax_event.axhline(level, color='dimgray', linestyle='--', linewidth=.6, alpha=.8, zorder=1)
    ax_event.axhline(0, color='dimgray', linewidth=1, zorder=2)

    ### Plot major El Niño events, JJA of year 1 (x = 0) to JJA of year 2 (x = 12)
    for y in colors:
        ax_event.plot(np.arange(13), series.loc[f'{y}-07-01':f'{y + 1}-07-01'].values, color=colors[y], linewidth=1.5, zorder=3)

    ### Plot current event
    current = series.loc[f'{year_now}-07-01':times[last]].values
    ax_event.plot(np.arange(len(current)), current, color=current_color, linewidth=3, zorder=5)
    ax_event.scatter(len(current) - 1, current[-1], s=40, color='gold', edgecolor='k', linewidth=.8, zorder=6)

    ### Ticks
    ax_event.set_xticks(np.arange(13), [rf'{s[0]}\textbf{{{s[1]}}}{s[2]}' for s in (seasons[(m - 1) % 12] for m in range(7, 20))], color='k')
    ax_event.set_yticks(np.arange(-1, 3.1, 1), [f'{y:.0f}' for y in np.arange(-1, 3.1, 1)], color='k', fontsize=11)
    ax_event.set_yticks(np.arange(-1, 3, .5), minor=True)
    ax_event.tick_params('both', length=5.5, width=2, which='major', color='dimgray', labelcolor='k', labelsize=9)
    ax_event.tick_params(axis='y', which='minor', length=3.5, width=1.5, color='dimgray')

    ### Domain
    ax_event.axis([0, 12, -1, 3])

    ### Title
    ax_event.set_title(rf'\textbf{{MAJOR EL NI\~NO EVENTS: {name}}}', color='k', fontsize=15, pad=6)

### Label
ax_oni.set_ylabel(r'\textbf{Index [°C]}', fontsize=15, color='k', labelpad=8)

### Legend for both bottom panels
l_handles = [plt.Line2D([0], [0], color=c, lw=1.5) for c in colors.values()] + [plt.Line2D([0], [0], color=current_color, lw=3)]
ax_roni.legend(l_handles, [event_label(y) for y in colors] + [event_label(year_now)],
               fontsize=9, loc='upper left', bbox_to_anchor=(1.01, 1), ncol=1, frameon=False, handlelength=1.6)

### Save figure
plt.savefig(os.path.join(fig_dir, f'Reanalysis_ENSO_ONI_RONI_Diff_Events_{year_min}–{year_max}.png'), dpi=1000)
plt.close()

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')