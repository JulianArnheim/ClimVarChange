"""
Plot Temperature (T) profile with Climate Change Indices (CCIs)
outlined as shown in Peings et al. (2017) and Arnheim et al. (2025)

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from matplotlib import patches
import matplotlib.colors as mcolors, matplotlib.pyplot as plt
import numpy as np
import os
import time
import xarray as xr

def cosine_weighted_spatial_mean(data, coord=None):
    coord = coord or ('lat' if 'lat' in data.coords else 'latitude' if 'latitude' in data.coords else None)
    if not coord:
        raise ValueError("No 'lat' or 'latitude' coordinate found in the data array.")
    if 'lat' in data.coords and 'latitude' in data.coords:
        raise ValueError("Both 'lat' and 'latitude' coordinates exist.")
    weights = np.cos(np.deg2rad(data[coord]))
    return (data * weights).sum(dim=coord) / (data.notnull() * weights).sum(dim=coord)

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Profile/Historical/Global/10 top/T/')

### Years
year_min, year_max = 1940, 2025

### Epoch
epoch_1 = (1981, 2010)

print(f'\n---- Atmospheric Temperature Profile ----\n')

### Load data
datat = []
for start_year in range(year_min, 2030, 10):
    end_year = min(start_year + 9, year_max)
    filename = os.path.join(data_dir, f'ERA5_Global_{start_year}-{end_year}_TProfile_Monthly.nc')
    if os.path.exists(filename):
        ds = xr.open_dataset(filename, chunks={'time': 12, 'latitude': 16, 'level': 10}).astype(np.float32)
        datat.append(ds)
        print(f'Loaded: {start_year}–{end_year}')
if datat:
    datat = xr.concat(datat, dim='time').sortby('time')
else:
    raise FileNotFoundError('No ERA5_Global_*_TProfile_Monthly.nc files found.')
print()

### Month settings
months_options = {'1. ANNUAL': list(range(1, 13)), '2. DJF': [12, 1, 2], '3. JJA': [6, 7, 8]}

### Loop through each set of months
for sel_month, mons in months_options.items():

    ### String corrections
    text_months = sel_month.split()[1]

    ### Figure
    print(f'{text_months}')

    ### Calculate and save annual zonal means
    full_year_min = year_min + 1 if text_months == 'DJF' else year_min
    annual_zonal_mean_datat_filename = os.path.join(data_dir, f'ERA5_Global_{full_year_min}-{year_max}_TProfile_{text_months}_NH_AnnualZonalMeans.nc')
    if os.path.exists(annual_zonal_mean_datat_filename):
        annual_zonal_mean_datat = xr.open_dataset(annual_zonal_mean_datat_filename)
        print(f'Using Previously Computed ERA5 Annual Zonal Means')

    else:
        ### Month specification
        datat_season = datat.sel(time=datat['time'].dt.month.isin(mons))

        ### Process seasonal mean
        if text_months == 'DJF':
            winter_year = xr.where(datat_season['time'].dt.month == 12, datat_season['time'].dt.year + 1, datat_season['time'].dt.year)
            annual_zonal_mean_datat = datat_season.mean(dim='longitude').assign_coords(winter_year=('time', winter_year.data))
            annual_zonal_mean_datat = annual_zonal_mean_datat.groupby('winter_year').mean(dim='time')
            annual_zonal_mean_datat = annual_zonal_mean_datat.sel(winter_year=slice(full_year_min, year_max)).rename({'winter_year': 'year'})
        else:
            annual_zonal_mean_datat = datat_season.mean(dim='longitude').groupby('time.year').mean(dim='time')

        ### Save processed data
        annual_zonal_mean_datat.to_netcdf(annual_zonal_mean_datat_filename)
        print(f'Saved {annual_zonal_mean_datat_filename}, the ERA5 Annual Zonal Means\n')

    ### Mean profile
    mean_profile = annual_zonal_mean_datat.sel(year=slice(epoch_1[0], epoch_1[1])).mean(dim='year')

    ### Figure design
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
    plt.rc('savefig', facecolor='w')
    plt.rc('axes', edgecolor='dimgray', facecolor='w', labelcolor='k')
    plt.rc('xtick', color='dimgray', labelcolor='k')
    plt.rc('ytick', color='dimgray', labelcolor='k')

    ### Create figure
    fig, ax = plt.subplots(figsize=(9, 5))

    ### Spines
    for side in ['top', 'right', 'bottom', 'left']:
        ax.spines[side].set_color('dimgray')
        ax.spines[side].set_linewidth(2)
        ax.spines[side].set_zorder(10)

    ### Ticks
    xticks = np.arange(-90, 91, 15)
    zscale = np.array([1000, 700, 500, 300, 200, 100, 50, 30, 10])
    ax.tick_params(axis='both', direction='out', which='major', pad=3, width=2, color='dimgray')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.yscale('log')
    plt.minorticks_off()
    plt.gca().invert_yaxis()
    ax.invert_yaxis()
    ax.set_xticks(xticks)
    ax.set_yticks(zscale)

    ### Tick labels
    ax.set_xticklabels([f'{tick}°' for tick in xticks], fontname='Avant Garde', fontsize=11)
    ax.set_yticklabels([str(p) for p in zscale], fontname='Avant Garde', fontsize=11)

    ### Domain
    ax.set(xlim=[-90, 90], ylim=[1000, 10])

    ### Plot
    levels = np.arange(-90, 31, 5)
    norm = mcolors.TwoSlopeNorm(vmin=-90, vcenter=0, vmax=30)
    im = ax.contourf(mean_profile['latitude'], mean_profile['level'], mean_profile.t, levels=levels, cmap=plt.get_cmap('RdBu_r'), norm=norm, extend='both', zorder=0)

    ### Colorbar
    ticks = np.arange(-90, 31, 10)
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', aspect=40, pad=.15, shrink=.8)
    cbar.ax.tick_params(labelsize=12, labelright=True, labelleft=False, right=True, left=False, color='dimgray')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(i) for i in ticks])
    cbar.ax.minorticks_off()
    cbar.ax.set_xlabel(r'\textbf{Temperature [°C]}', fontsize=16, color='k', labelpad=6)

    ### Title
    plt.title(r'\textbf{TEMPERATURE $(T)$}', color='k', fontsize=20, y=1.04)

    ### Labels
    plt.xlabel(r'\textbf{Latitude [°N]}', color='k', fontsize=15)
    plt.ylabel(r'\textbf{Pressure [hPa]}', color='k', fontsize=15)

    ### Annotation
    season_prefix = '' if text_months == 'ANNUAL' else f'{text_months} '
    plt.text(1, 1.05, rf'\textbf{{{season_prefix}{epoch_1[0]}–{epoch_1[1]}}}', fontsize=10, rotation='horizontal', ha='right', va='top', color='k', transform=ax.transAxes)

    ### Save figure
    plt.subplots_adjust(bottom=.05)
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_{epoch_1[0]}–{epoch_1[1]}_TProfile_10top_{text_months}.png'), dpi=1000)

    ### Annotations
    ### UTW
    ax.add_patch(plt.Rectangle((-20, 150), 40, 250, edgecolor='k', facecolor='none', linewidth=2, zorder=5, label='UTW'))
    ax.annotate('UTW', (0, 250), color='k', fontsize=10, ha='center', va='center', weight='bold')
    utw_temps = mean_profile.sel(latitude=slice(20, -20), level=slice(400, 150))
    utw_avg_temps = cosine_weighted_spatial_mean(utw_temps).t.mean().values
    print(f'UTW: {utw_avg_temps:.1f}°C')

    ### AW
    ax.add_patch(plt.Rectangle((60, 700), 30, 300, edgecolor='k', facecolor='none', zorder=5, linewidth=2))
    ax.annotate('AW', (75, 850), color='k', fontsize=10, ha='center', va='center', weight='bold')
    aw_temps = mean_profile.sel(latitude=slice(90, 60), level=slice(1000, 700))
    aw_avg_temp = cosine_weighted_spatial_mean(aw_temps).t.mean().values
    print(f'AW: {aw_avg_temp:.1f}°C')

    ### RUTAW
    arrow_1 = (20, 400)
    arrow_2, arrow_angle = (60, 700), -12
    arrow = patches.FancyArrowPatch(arrow_2, arrow_1, connectionstyle='arc3,rad=0.0', arrowstyle='<->', mutation_scale=15, zorder=5, color='k')
    ax.add_patch(arrow)
    arrow_midpoint = ((arrow_2[0] + arrow_1[0]) / 2, ((arrow_2[1] + arrow_1[1]) / 2) - 20)
    ax.annotate('RUTAW (Ratio of Changes)', xy=arrow_midpoint, color='k', fontsize=7, ha='center', va='bottom', weight='bold', rotation=arrow_angle,
                rotation_mode='anchor', xytext=(np.cos(np.deg2rad(arrow_angle)), 1 + np.sin(np.deg2rad(arrow_angle))), textcoords='offset points')

    ### PST
    ax.add_patch(plt.Rectangle((65, 30), 25, 220, edgecolor='k', facecolor='none', zorder=5, linewidth=2))
    ax.annotate('PST', (78, 95), color='k', fontsize=10, ha='center', va='center', zorder=5, weight='bold')
    pst_temps = mean_profile.sel(latitude=slice(90, 65), level=slice(250, 30))
    pst_avg_temps = cosine_weighted_spatial_mean(pst_temps).t.mean().values
    print(f'PST: {pst_avg_temps:.1f}°C')

    ### UPTG
    u_trop = mean_profile.sel(latitude=slice(20, -20), level=slice(400, 150))
    u_trop = cosine_weighted_spatial_mean(u_trop).t.mean().values
    u_arctic = mean_profile.sel(latitude=slice(90, 65), level=slice(400, 150))
    u_arctic = cosine_weighted_spatial_mean(u_arctic).t.mean().values
    uptg = u_trop - u_arctic
    print(f'UPTG: {uptg:.1f}°C')

    ### LOTG
    l_trop = mean_profile.sel(latitude=slice(20, -20), level=slice(1000, 700))
    l_trop = cosine_weighted_spatial_mean(l_trop).t.mean().values
    l_arctic = mean_profile.sel(latitude=slice(90, 65), level=slice(1000, 700))
    l_arctic = cosine_weighted_spatial_mean(l_arctic).t.mean().values
    lotg = l_trop - l_arctic
    print(f'LOTG: {lotg:.1f}°C')

    ### Save figure with annotations
    plt.subplots_adjust(bottom=.05)
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_{epoch_1[0]}–{epoch_1[1]}_TProfile_10top_{text_months}_annotated.png'), dpi=1000)

    ### Patches for dotted annotations
    edge_properties = {'edgecolor': 'k', 'facecolor': 'none', 'linewidth': 2, 'linestyle': 'dotted'}
    ax.add_patch(plt.Rectangle((-20, 150), 40, 250, **edge_properties))
    ax.add_patch(plt.Rectangle((65, 150), 25, 250, **edge_properties))
    ax.add_patch(plt.Rectangle((-20, 700), 40, 300, **edge_properties))
    ax.add_patch(plt.Rectangle((65, 700), 25, 300, **edge_properties))

    ### Arrow for upper tropospheric temperature gradient
    arrow_1 = (20, 300)
    arrow_2, arrow_angle = (65, 300), 0
    ax.add_patch(patches.FancyArrowPatch(arrow_2, arrow_1, connectionstyle='arc3,rad=0.0', arrowstyle='<->', mutation_scale=15, color='k'))
    arrow_midpoint = ((arrow_2[0] + arrow_1[0]) / 2, (arrow_2[1] + arrow_1[1]) / 2)
    ax.annotate('UPTG (Difference)', xy=arrow_midpoint, color='k', fontsize=7, ha='center', va='bottom', weight='bold', rotation=arrow_angle,
        rotation_mode='anchor', xytext=(np.cos(np.deg2rad(arrow_angle)), 1 + np.sin(np.deg2rad(arrow_angle))), textcoords='offset points')

    ### Arrow for lower tropospheric temperature gradient
    arrow_1, arrow_2 = (20, 875), (65, 875)
    ax.add_patch(patches.FancyArrowPatch(arrow_2, arrow_1, connectionstyle='arc3,rad=0.0', arrowstyle='<->', mutation_scale=15, color='k'))
    arrow_midpoint = ((arrow_2[0] + arrow_1[0]) / 2, (arrow_2[1] + arrow_1[1]) / 2)
    ax.annotate('LOTG (Difference)', xy=arrow_midpoint, color='k', fontsize=7, ha='center', va='bottom', weight='bold', rotation=arrow_angle,
        rotation_mode='anchor', xytext=(np.cos(np.deg2rad(arrow_angle)), 1 + np.sin(np.deg2rad(arrow_angle))), textcoords='offset points')

    ### Save figure with annotations and dotted annotations
    plt.subplots_adjust(bottom=.05)
    plt.savefig(os.path.join(fig_dir, f'Reanalysis_{epoch_1[0]}–{epoch_1[1]}_TProfile_10top_{text_months}_annotated_dotted.png'), dpi=1000)
    plt.close()

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')