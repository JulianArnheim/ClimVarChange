"""
Plot Climate Change Indices (CCIs) from ERA5 reanalysis

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
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks_position('bottom')

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Timeseries/CCIs/')

### Years and epoch
year_min, year_max, epoch = 1940, 2025, (1981, 2010)

print(f'\n---- Climate Change Indices (CCIs) ----\n')

### Specifications
regions, season_options = ['UTW', 'AA', 'PST', 'UPTG', 'LOTG'], ['1. ANNUAL', '2. DJF', '3. JJA']
sector_options = {'NH': 'Northern Hemisphere (180°W–180°E)', 'AM': 'American Sector (140°W–50°W)', 'AT': 'Atlantic Sector (50°W–40°E)',
                  'AS': 'Asian Sector (40°E–130°E)', 'PA': 'Pacific Sector (130°E–140°W)'}

### Load data
datat = xr.open_dataset(os.path.join(data_dir, f'ERA5_CCIs_{year_min}-{year_max}_Annual.nc'))

### Loop through anomalies and total
for anomaly in [True, False]:

    ### Loop through each season and sector
    for season_name in season_options:

        ### String corrections
        text_months = season_name.split()[1]
        season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '

        for sector_key, sector_name in sector_options.items():

            ### Figure
            print(f'{text_months} | {sector_key} | Anomaly: {anomaly}')

            ### Fresh copy of data, compute anomaly
            CCI_datat = datat.copy()
            if anomaly:
                baseline_mask = (CCI_datat['year'] >= epoch[0]) & (CCI_datat['year'] <= epoch[1])
                for r in regions:
                    var = f'ERA5_{text_months}_{sector_key}_{r}_CCIs'
                    CCI_datat[var] = CCI_datat[var] - CCI_datat[var].where(baseline_mask).mean(dim='year')

            ### Figure design
            plt.rc('text', usetex=True)
            plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
            plt.rc('savefig', facecolor='w')
            plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='dimgray')
            plt.rc('xtick', color='dimgray')
            plt.rc('ytick', color='dimgray')

            ### Letters
            letters_text = ['(a)', '(b)', '(c)', '(d)', '(e)']
            for letters in [False, True]:

                ### Create figure
                fig, axes = plt.subplots(len(regions), 1, figsize=(7, 7))
                for i, r in enumerate(regions):

                    ### Subplot
                    ax = axes[i]
                    ax.set_title('AW' if r == 'AA' else r, y=.92, fontsize=12)
                    var = f'ERA5_{text_months}_{sector_key}_{r}_CCIs'

                    ### Plot
                    years, CCI_vals = CCI_datat['year'].values, CCI_datat[var].values
                    ax.plot(years, CCI_vals, color='k', linewidth=1)

                    ### LOWESS smoothing
                    num_lowess = 10  # window in years
                    lowess_smoothed = lowess(CCI_vals, years, frac=(num_lowess / (years[-1] - years[0])), return_sorted=False)
                    ax.plot(years, lowess_smoothed, color='deeppink', linewidth=1.5)

                    ### Adjust axes
                    adjust_spines(ax, ['left', 'bottom'])
                    ax.spines['bottom'].set_linewidth(2)
                    ax.spines['left'].set_linewidth(2)
                    ax.tick_params('both', length=4.5, width=2, which='major', color='dimgray', labelsize=10)

                    ### Ticks and domain
                    ax.set_xticks(np.arange(1940, 2031, 10))
                    ax.set_xticklabels(ax.get_xticks(), fontsize=9, color='k', fontname='Avant Garde')
                    min_y, max_y = int(np.floor(CCI_datat[var].min())), int(np.ceil(CCI_datat[var].max()))
                    ytick_step = 4 if max_y - min_y > 10 else 2
                    min_y -= min_y % ytick_step
                    max_y += (-max_y) % ytick_step
                    ytick_vals = np.arange(min_y, max_y + 1, ytick_step)
                    ax.set_yticks(ytick_vals)
                    ax.set_yticklabels([f'+{y:.0f}' if anomaly and y > 0 else f'{y:.0f}' for y in ytick_vals], fontsize=9, color='k', fontname='Avant Garde')
                    ax.axis([1940, 2030, min_y, max_y])

                    ### Horizontal lines
                    for y in range(min_y, max_y + 1):
                        ax.axhline(y=y, linestyle='-', color='gray', alpha=.5, linewidth=.5, zorder=0)

                ### Title and labels
                fig.suptitle(rf'\textbf{{Climate Change Indices: {season_prefix}{sector_name}}}', fontsize=15, y=.99)
                fig.text(.01, .5, r'\textbf{Temperature [°C]}', va='center', rotation='vertical', fontsize=14, color='k', fontname='Avant Garde')
                axes[-1].set_xlabel(r'\textbf{Year}', fontsize=13, fontweight='bold', color='k', fontname='Avant Garde', labelpad=8)
                fig.text(.94, .942, 'Annual Mean', color='k', fontsize=6, ha='right', va='top', fontname='Avant Garde')
                fig.text(.94, .93, rf'\textbf{{{num_lowess}}}-Year LOWESS', color='deeppink', fontsize=6, ha='right', va='top', fontname='Avant Garde')

                plt.tight_layout(rect=[.045, -.02, .995, 1.01])

                ### Save figure
                save_dir = os.path.join(fig_dir, 'letters' if letters else 'no letters', *('anom',) if anomaly else ())
                suffix = ('_anom' if anomaly else '') + ('_letters' if letters else '')
                if letters:
                    for ax, ltr in zip(axes, letters_text):
                        ax.text(-.07, 1.12, rf'\textbf{{{ltr}}}', ha='left', va='bottom', color='k', fontsize=12, transform=ax.transAxes)
                fig_path = os.path.join(save_dir, f'Reanalysis_CCIs_{text_months}_{sector_key}_{year_min}–{year_max}{suffix}.png')
                plt.savefig(fig_path, dpi=1000)
                plt.close()

            ### Progress
            print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')