"""
Plot Sinuosity Index (SIN) values from Cattiaux et al. (2016),
Peings et al. (2017, 2018), Arnheim et al. (2025) for reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from scipy.stats import gamma, kstest
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

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Histograms/SIN/')

### Years
year_min, year_max = 1981, 2010

print(f'\n---- Sinuosity Index (SIN) ----\n')

### Decade range for looping
decade_floor = (year_min // 10) * 10
decade_ceiling = year_max + 11

### Load data
datasets = []
for start_year in range(decade_floor, decade_ceiling, 10):
    end_year = min(start_year + 9, year_max + 11)
    filename = os.path.join(data_dir, f'ERA5_SIN_Sectors_{start_year}-{end_year}_Daily.nc')
    if os.path.exists(filename):
        datasets.append(xr.open_dataset(filename))
        print(f'Loaded: {start_year}–{end_year}')
SIN = xr.concat(datasets, dim='time')
print(f'Concatenated SINs\n')

### Process data
SIN = SIN.sel(time=slice(f'{year_min}-01-01', f'{year_max}-12-31'))['SIN_NH']

### Figure design
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
plt.rc('savefig', facecolor='w')
plt.rc('axes', edgecolor='dimgray')
plt.rc('xtick', color='dimgray')
plt.rc('ytick', color='dimgray')

### Create figure
fig, ax = plt.subplots(figsize=(6.5, 3))

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

### Title, ticks, label, and domain
plt.title(rf'\textbf{{Sinuosity Index (SIN) {year_min}–{year_max}}}', color='k', fontsize=20, y=1.02)
plt.xticks(np.arange(0, 3.25, .25), color='k')
plt.yticks(np.arange(0, 1000, 50), color='k')
plt.ylabel('Days', fontsize=14, labelpad=12)
plt.xlim([1, 3])
plt.ylim([0, 300])

### Dates for highest and lowest sinuosity
high_val, low_val = round(float(SIN[SIN.argmax()]), 3), round(float(SIN[SIN.argmin()]), 3)
high_date, low_date = str(SIN.time[SIN.argmax()].values)[:10], str(SIN.time[SIN.argmin()].values)[:10]
print(f'Highest SIN: {high_date}, at {high_val}')
print(f'Lowest SIN: {low_date}, at {low_val}\n')

### Find best fit Gamma distribution
SIN = SIN.values
SIN_max = np.max(SIN)  # candidate gamma locations below the minimum SIN
best_x0, best_alpha, best_theta, best_stat = None, None, None, np.inf
for x0 in np.arange(1, SIN_max, .001):
    ### Skip invalid loc values
    if x0 >= SIN.min():
        continue
    try:
        ### Fit with this fixed loc
        alpha, _, theta = gamma.fit(SIN, floc=x0)
        ### Kolmogorov-Smirnov test
        D_stat, _ = kstest(SIN, 'gamma', args=(alpha, x0, theta))
        print(f'x0={x0:.3f}, alpha={alpha:.3f}, theta={theta:.3f}, KS={D_stat:.5f}')
        ### Track best fit
        if D_stat < best_stat:
            best_x0, best_alpha, best_theta, best_stat = x0, alpha, theta, D_stat
    except Exception as e:
        print(f'x0={x0:.3f} skipped due to error: {e}')
print(f'\nBest fit: x0={best_x0:.3f}, alpha={best_alpha:.3f}, theta={best_theta:.3f}, KS={best_stat:.5f}')

### Histogram
bins = np.arange(1, 3.01, .01)
hist_vals, edges = np.histogram(SIN, bins=bins)
bin_width = edges[1] - edges[0]
plt.bar(edges[:-1], hist_vals, width=bin_width, color='r', edgecolor='none', zorder=1)

### Probability density function
x_vals = np.linspace(1, 3, 1000)
pdf_scaled = gamma.pdf(x_vals, a=best_alpha, loc=best_x0, scale=best_theta) * len(SIN) * bin_width
plt.plot(x_vals, pdf_scaled, color='k', linewidth=1, linestyle='--', label='Gamma Fit', zorder=2)

### Legend and annotations
plt.legend(loc='upper center', frameon=False, fontsize=9)
plt.text(1.68, 200, r'$f(x) = \frac{(x - x_0)^{\alpha - 1} e^{-\frac{(x - x_0)}{\theta}}}{\theta^\alpha \, \Gamma(\alpha)}$', fontsize=14, color='k')
plt.text(1.9, 100, f'$\\alpha = {best_alpha:.2f}$\n$\\theta = {best_theta:.2f}$\n$x_0 = {best_x0:.2f}$', fontsize=10, color='k')

### Assess fit with a parametric bootstrap of the KS statistic, since fitted parameters make the standard KS p-value too lenient
rng = np.random.default_rng(0)
D_stat = kstest(SIN, 'gamma', args=(best_alpha, best_x0, best_theta))[0]
D_boot = [kstest(s, 'gamma', args=gamma.fit(s, floc=best_x0))[0] for s in (gamma.rvs(best_alpha, best_x0, best_theta, size=SIN.size, random_state=rng) for _ in range(200))]
p_value = np.mean(np.array(D_boot) >= D_stat)
print(f'  Kolmogorov-Smirnov Statistic: {D_stat:.5f}')
print(f'  Bootstrap p-value (conditional on x0): {p_value:.3f}\n')

### Save figure
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'Reanalysis_SIN_{year_min}–{year_max}.png'), dpi=1000)

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')