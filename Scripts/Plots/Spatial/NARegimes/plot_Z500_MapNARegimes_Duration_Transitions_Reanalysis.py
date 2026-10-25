"""
Plot 500 hPa geopotential height (Z500) anomaly clusters in the wintertime North Atlantic from
ERA5 reanalysis based on Arnheim et al (2025) plus a persistence minimum (Falkena et al. 2020)

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
from scipy.stats import gamma
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import xarray as xr

def adjust_spines(ax, spines, outward=3, color='dimgray', lw=2):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', outward))
            spine.set_linewidth(lw)
            spine.set_color(color)
        else:
            spine.set_color('none')
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
    else:
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_ticks([])

def calculate_regime_durations(labels, times):
    durations = {cluster: [] for cluster in cluster_order}
    current_regime, duration = labels[0], 1
    for i in range(1, len(labels)):
        if labels[i] == current_regime and times[i] - times[i - 1] == np.timedelta64(1, 'D'):  # runs end at gaps between winters
            duration += 1
        else:
            durations[current_regime].append(duration)
            current_regime, duration = labels[i], 1
    durations[current_regime].append(duration)
    return durations

def calculate_transition_matrix(labels, times):
    matrix = np.zeros((4, 4))
    for i in range(1, len(labels)):
        ### Exclude transitions across gaps between winters
        if times[i] - times[i - 1] != np.timedelta64(1, 'D'):
            continue
        current_idx = cluster_order.index(labels[i - 1])
        next_idx = cluster_order.index(labels[i])
        ### Count transitions between different regimes
        if current_idx != next_idx:
            matrix[current_idx, next_idx] += 1
    ### Normalize each row to percentages
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0) * 100
    return matrix

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/NA/Regimes/')

print(f'\n---- Weather Regime Clusters ----\n')

### Years and epochs
year_min, year_max = 1940, 2025
epochs = [(year_min, year_max), (1981, 2010)]

### Cluster specifications
cluster_colors, cluster_order = {'NAO+': 'r', 'BL': 'royalblue', 'NAO-': 'g', 'AR': 'purple'}, ['NAO+', 'BL', 'NAO-', 'AR']

### Loop through epochs
for start_year, end_year in epochs:

    ### Seasons
    seasons = {'DJF': [12, 1, 2], 'DJFM': [12, 1, 2, 3]}

    ### Loop through seasons
    for season_name, season_months in seasons.items():

        ### First valid year
        first_year = start_year + 1 if (start_year == year_min) else start_year

        print(f'{season_name} | {start_year}–{end_year}')

        ### Paths for saved files
        cluster_filename = os.path.join(data_dir, f'ERA5_NA_Z500_Clustered_Regimes_{season_name}_{first_year}-{end_year}.nc')
        cluster_ds_filenames = [os.path.join(data_dir, f'ERA5_NA_Cluster_{cluster}_{season_name}_{first_year}-{end_year}.nc') for cluster in cluster_order]

        ### Load cluster data
        cluster_data = xr.open_dataset(cluster_filename)
        saved_order = [s.strip() for s in cluster_data.attrs.get('cluster_order', '').split(',')]
        cluster_frequencies, times = cluster_data['cluster_frequencies'].values, cluster_data['labels'].time.values
        labels = np.array([saved_order[i] for i in cluster_data['labels'].values.astype(int)])  # regime labels

        ### Duration statistics and transition matrix
        durations, transition_matrix = calculate_regime_durations(labels, times), calculate_transition_matrix(labels, times)

        ### Load cluster datasets in plotting order
        loaded_datasets = {name: xr.open_dataset(f) for name, f in zip(saved_order, cluster_ds_filenames)}
        cluster_datasets = [loaded_datasets[name] for name in cluster_order]

        ### Figure design
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
        plt.rc('savefig', facecolor='w')
        plt.rc('xtick', color='k')
        plt.rc('ytick', color='k')
        plt.rc('axes', labelcolor='k')
        plt.rc('axes', facecolor='w')

        ### Clusters
        cluster_info = {'NAO+': {'short': 'NAO+', 'long': 'Positive NAO (NAO+)', 'color': 'r'},
                          'BL': {'short': 'BL', 'long': 'European Blocking (BL)', 'color': 'royalblue'},
                        'NAO-': {'short': 'NAO-', 'long': 'Negative NAO (NAO-)', 'color': 'g'},
                          'AR': {'short': 'AR', 'long': 'Atlantic Ridge (AR)', 'color': 'purple'}}

        ### Create figure
        fig = plt.figure(figsize=(18, 5))
        outer_gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=.22, top=.75, bottom=.1)
        maps_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[0], wspace=.12, hspace=.13)
        pdf_gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[1], wspace=.26, hspace=.4)
        matrix_gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[2])

        ### Create axes
        map_axes = [fig.add_subplot(maps_gs[0, 0], projection=ccrs.Robinson(central_longitude=0)), fig.add_subplot(maps_gs[0, 1], projection=ccrs.Robinson(central_longitude=0)),
                    fig.add_subplot(maps_gs[1, 0], projection=ccrs.Robinson(central_longitude=0)), fig.add_subplot(maps_gs[1, 1], projection=ccrs.Robinson(central_longitude=0))]
        pdf_axes = [fig.add_subplot(pdf_gs[0, 0]), fig.add_subplot(pdf_gs[0, 1]), fig.add_subplot(pdf_gs[1, 0]), fig.add_subplot(pdf_gs[1, 1])]
        matrix_ax = fig.add_subplot(matrix_gs[0])

        ### Subplot titles
        maps_bbox, pdf_bbox, matrix_bbox = (outer_gs[0].get_position(fig), outer_gs[1].get_position(fig), outer_gs[2].get_position(fig))
        fig.text((maps_bbox.x0 + maps_bbox.x1) / 2, maps_bbox.y1 + .07, r'\textbf{Weather Regimes}', fontsize=13, ha='center', va='bottom')
        fig.text((pdf_bbox.x0 + pdf_bbox.x1) / 2, pdf_bbox.y1 + .07, r'\textbf{Duration}', fontsize=13, ha='center', va='bottom')
        fig.text((matrix_bbox.x0 + matrix_bbox.x1) / 2 - .02, matrix_bbox.y1 + .07, r'\textbf{Transitions}', fontsize=13, ha='center', va='bottom')

        #########################
        ### Left panel : Maps ###
        #########################
        letter_idx, letters = 0, [chr(i) for i in range(ord('a'), ord('z') + 1)]
        for idx, (ax, cluster_ds) in enumerate(zip(map_axes, cluster_datasets)):
            ax.coastlines(linewidth=.5, resolution='50m', color='saddlebrown')
            ax.add_feature(cfeature.RIVERS.with_scale('50m'), edgecolor='saddlebrown', facecolor='none', linewidth=.5)
            ax.set_extent([-90, 30, 20, 80], crs=ccrs.PlateCarree())

            ### Plot
            x, y = np.meshgrid(cluster_ds['longitude'], cluster_ds['latitude'])
            cf = ax.contourf(x, y, cluster_ds['anomaly'].values, levels=np.linspace(-250, 250, 21), cmap=plt.get_cmap('RdBu_r', lut=192),
                             norm=plt.Normalize(vmin=-250, vmax=250), transform=ccrs.PlateCarree())

            ### Contours
            cs = ax.contour(x, y, cluster_ds['Z500'].values, levels=np.arange(4500, 6500, 100), colors='k', linewidths=.5, transform=ccrs.PlateCarree())
            ax.clabel(cs, cs.levels, inline=True, fmt='%d', fontsize=5)

            ### Letters
            ax.text(.06, 1.18, rf'\textbf{{({letters[letter_idx]})}}', fontsize=11, ha='right', va='top', transform=ax.transAxes)
            letter_idx += 1

            ### Title
            color = cluster_info[cluster_order[idx]]['color']
            ax.set_title(rf"\textbf{{{cluster_info[cluster_order[idx]]['short']}}}", fontsize=10, pad=5, color=color)

            ### Colored borders
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.25)
                spine.set_zorder(5)

            ### Regime frequency
            ax.text(.995, 1.1, rf'{cluster_frequencies[idx]:.1f}\%', fontsize=7, ha='right', va='top', transform=ax.transAxes)

        ### Map colorbar
        cbar = fig.colorbar(cf, ax=map_axes, orientation='horizontal', aspect=40, shrink=.9, pad=.08)
        ticks = np.linspace(-250, 250, 11)
        cbar.set_ticks(ticks)
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.set_xticklabels([f'{int(i):+d}' if i != 0 else '0' for i in ticks])
        cbar.ax.set_xlabel(r'\textbf{500 hPa Geopotential Height Anomaly [gpm]}', fontsize=9, labelpad=5)

        ####################################
        ### Middle panel : Duration PDFs ###
        ####################################
        gamma_params, x_fit = {}, np.linspace(0, 25, 400)
        for idx, ax in enumerate(pdf_axes):
            regime, x0 = cluster_order[idx], 2.5  # persistence minimum (3 days) less a half-day continuity correction
            data = np.asarray(durations[regime])
            data = data[data > x0]  # same subset for the histogram and the fit

            ### Histogram
            ax.hist(data, bins=np.arange(.5, 25.5, 1), density=True, color=cluster_colors[regime], edgecolor='k', linewidth=.5)

            ### Gamma fit shifted by x0
            alpha, _, theta = gamma.fit(data - x0, floc=0)
            gamma_params[regime] = (alpha, theta)
            pdf = gamma.pdf(x_fit - x0, a=alpha, loc=0, scale=theta)
            pdf[x_fit < x0] = np.nan
            ax.plot(x_fit, pdf, 'k--', lw=1.6)

            ### Ticks and domain
            xticks, yticks = np.arange(0, 26, 5), np.arange(0, .26, .05)
            ax.set_xticks(xticks); ax.set_yticks(yticks)
            ax.set_xticklabels([f'{x:d}' for x in xticks], fontsize=8); ax.set_yticklabels(['0' if y == 0 else f'{y:.2f}'.lstrip('0').rstrip('.') for y in yticks], fontsize=7)
            ax.set_xlim(0, 25); ax.set_ylim(0, .25)

            ### Letters
            ax.text(-.19, 1.17, rf'\textbf{{({letters[letter_idx]})}}', transform=ax.transAxes, fontsize=11, ha='left', va='top')
            letter_idx += 1

            ### Figure design
            adjust_spines(ax, ['left', 'bottom'], outward=3, lw=2)
            ax.xaxis.tick_bottom(), ax.yaxis.tick_left()
            ax.tick_params(axis='both', which='major', length=4, width=2, color='dimgray', labelsize=7)

            ### Title and labels
            ax.set_title(rf"\textbf{{{cluster_info[regime]['long']}}}", fontsize=10, color=cluster_colors[regime])
            if idx in [2, 3]:
                ax.set_xlabel(r'\textbf{Duration [days]}', fontsize=9)

            ### Gamma fit annotations
            gamma_vars = f'$\\alpha$ = {alpha:.1f}\n$\\theta$ = {theta:.1f}'
            ax.text(.9, .93, gamma_vars, transform=ax.transAxes, ha='right', va='top', fontsize=9, bbox=dict(facecolor='w', edgecolor='none', alpha=1))
            ax.text(.9, .73, f'$x_0$ = 3.0', transform=ax.transAxes, ha='right', va='top', fontsize=9, color='gray')
            ax.text(.9, .5, rf'$\mu$ = $\alpha$$\theta$ = {alpha * theta:.1f}', transform=ax.transAxes, ha='right', va='top', fontsize=9, color='k')

        #######################################
        ### Right panel : Transition Matrix ###
        #######################################
        levels = np.arange(0, 51, 5)
        tm = transition_matrix.copy()
        np.fill_diagonal(tm, np.nan)

        ### Colormap
        cmap = plt.get_cmap('viridis', len(levels)).copy()
        cmap.set_bad(color='lightgray')
        im = matrix_ax.imshow(np.ma.masked_invalid(tm), cmap=cmap, norm=plt.matplotlib.colors.BoundaryNorm(boundaries=levels, ncolors=len(levels), extend='max'))

        ### Gridlines
        matrix_ax.set_xticks(np.arange(-.5, 4, 1), minor=True); matrix_ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
        matrix_ax.grid(which='minor', color='w', linestyle='-', linewidth=1.25)
        matrix_ax.tick_params(axis='both', which='major', length=0, pad=8); matrix_ax.tick_params(axis='both', which='minor', length=0, pad=8)
        matrix_ax.set_frame_on(False)

        ### Labels
        matrix_ax.set_xticks(np.arange(4)); matrix_ax.set_yticks(np.arange(4))
        matrix_ax.set_xticklabels([rf"\textbf{{{cluster_info[r]['short']}}}" for r in cluster_order], fontsize=10)
        matrix_ax.set_yticklabels([rf"\textbf{{{cluster_info[r]['short']}}}" for r in cluster_order], fontsize=10)
        matrix_ax.set_xlabel(r'\textbf{Next Regime}', fontsize=11, labelpad=8)
        for tick, regime in zip(matrix_ax.get_xticklabels(), cluster_order):
            tick.set_color(cluster_colors[regime])
        for tick, regime in zip(matrix_ax.get_yticklabels(), cluster_order):
            tick.set_color(cluster_colors[regime])

        ### Percentages
        for i in range(4):
            for j in range(4):
                value = transition_matrix[i, j]
                if i == j:
                    continue
                matrix_ax.text(j, i, f'{value:.1f}', ha='center', va='center', fontsize=10, color='k' if value > 50 else 'w')

        ### Transition colorbar
        tick_levels = np.arange(0, 51, 10)
        matrix_cbar = fig.colorbar(im, ax=matrix_ax, orientation='vertical', shrink=.9, aspect=25, pad=.03, ticks=tick_levels, extend='max')
        matrix_cbar.ax.set_yticklabels([str(i) for i in tick_levels], fontsize=7)
        matrix_cbar.ax.tick_params(axis='y', which='major', labelsize=7); matrix_cbar.ax.tick_params(axis='y', which='minor', left=False, right=False)
        matrix_cbar.ax.set_ylabel(r'\textbf{Transition Frequency [\%]}', rotation=270, fontsize=9, labelpad=12)

        ### Letter
        matrix_ax.text(-.18, 1.12, rf'\textbf{{({letters[letter_idx]})}}', transform=matrix_ax.transAxes, fontsize=11, ha='left', va='top')

        ### Title and annotation
        fig.suptitle(rf'\textbf{{{season_name} North Atlantic Weather Regimes}}', fontsize=18, y=.99)
        fig.text(.5, .9, rf'\textbf{{{first_year}–{end_year}}}', fontsize=14, ha='center')

        ### Save figure
        plt.savefig(os.path.join(fig_dir, f'Reanalysis_Z500_NA_Regimes_Duration_Transitions_{season_name}_{first_year}–{end_year}.png'), dpi=1000, bbox_inches='tight')
        plt.close(fig)

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')