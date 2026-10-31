"""
Plot 500 hPa geopotential height (Z500) anomaly clusters in the wintertime North Atlantic from
ERA5 reanalysis based on Arnheim et al (2025) plus a persistence minimum (Falkena et al. 2020)

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import cartopy.crs as ccrs, cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import xarray as xr

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Spatial/NA/Regimes/')

### Epochs
epochs = [(1940, 2025), (1981, 2010)]

### Clusters
cluster_order = ['NAO+', 'BL', 'NAO-', 'AR']

print(f'\n---- Weather Regime Clusters ----\n')

### Loop through epochs
for start_year, end_year in epochs:

    ### Seasons
    seasons = {'DJF': [12, 1, 2], 'DJFM': [12, 1, 2, 3]}

    ### Loop through seasons
    for season_name, season_months in seasons.items():

        ### First valid year
        first_year = start_year + 1 if (start_year == 1940) else start_year

        print(f'{season_name} | {start_year}–{end_year}')

        ### Paths for saved files
        cluster_filename = os.path.join(data_dir, f'ERA5_NA_Z500_Clustered_Regimes_{season_name}_{first_year}-{end_year}.nc')
        cluster_ds_filenames = [os.path.join(data_dir, f'ERA5_NA_Cluster_{cluster}_{season_name}_{first_year}-{end_year}.nc') for cluster in cluster_order]

        ### Load cluster datasets in plotting order
        cluster_data = xr.open_dataset(cluster_filename)
        saved_order = [s.strip() for s in cluster_data.attrs.get('cluster_order', '').split(',')]
        cluster_frequencies = cluster_data['cluster_frequencies'].values
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
        cluster_infos = {'color': {'NAO+': {'title': 'Positive NAO (NAO+)', 'color': 'r'}, 'BL': {'title': 'European Blocking (BL)', 'color': 'royalblue'},
                                   'NAO-': {'title': 'Negative NAO (NAO-)', 'color': 'g'}, 'AR': {'title': 'Atlantic Ridge (AR)', 'color': 'purple'}},
                         'black': {'NAO+': {'title': 'Positive NAO (NAO+)', 'color': 'k'}, 'BL': {'title': 'European Blocking (BL)', 'color': 'k'},
                                   'NAO-': {'title': 'Negative NAO (NAO-)', 'color': 'k'}, 'AR': {'title': 'Atlantic Ridge (AR)', 'color': 'k'}}}

        ### Loop through cluster designs
        for style_name, cluster_info in cluster_infos.items():

            ### Create figure
            fig = plt.figure(figsize=(6.5, 5))
            ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.Robinson(central_longitude=0))
            ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.Robinson(central_longitude=0))
            ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.Robinson(central_longitude=0))
            ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.Robinson(central_longitude=0))
            axs, letters = [ax1, ax2, ax3, ax4], ['(a)', '(b)', '(c)', '(d)']

            for idx, i in enumerate(range(len(cluster_order))):
                ax = axs[idx]
                ax.coastlines(linewidth=.5, resolution='50m', color='saddlebrown')
                ax.add_feature(cfeature.RIVERS.with_scale('50m'), edgecolor='saddlebrown', facecolor='none', linewidth=.5)
                ax.set_extent([-90, 30, 20, 80], crs=ccrs.PlateCarree())

                ### Plot
                cluster_ds = cluster_datasets[idx]
                cmap = plt.get_cmap('RdBu_r', lut=192)
                levels, norm = np.linspace(-250, 250, 21), plt.Normalize(vmin=-250, vmax=250)
                x, y = np.meshgrid(cluster_ds['longitude'], cluster_ds['latitude'])
                cf = ax.contourf(x, y, cluster_ds['anomaly'].values, levels=levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

                ### Contours
                levels = np.arange(4500, 6500, 100)
                cs = ax.contour(x, y, cluster_ds['Z500'].values, levels=levels, colors='k', linewidths=.5, transform=ccrs.PlateCarree())
                ax.clabel(cs, cs.levels, inline=True, fmt='%d', fontsize=5)

                ### Letter
                ax.text(.06, 1.14, rf'\textbf{{{letters[idx]}}}', fontsize=12, ha='right', va='top', transform=ax.transAxes)

                ### Name, color, label according to cluster
                cluster_name = cluster_info[cluster_order[idx]]['title']
                cluster_color = cluster_info[cluster_order[idx]]['color']
                ax.set_title(cluster_name, fontsize=11, pad=6, color=cluster_color)
                for spine in ax.spines.values():
                    spine.set_edgecolor(cluster_color)
                    spine.set_linewidth(1.25)
                    spine.set_zorder(5)
                ax.text(.995, 1.09, rf'{cluster_frequencies[i]:.1f}\%', fontsize=8, ha='right', va='top', transform=ax.transAxes)

            ### Colorbar
            ticks = np.linspace(-250, 250, 11)
            cbar = plt.colorbar(cf, orientation='horizontal', aspect=50, pad=.05, shrink=.9, ticks=ticks)
            cbar.ax.tick_params(labelsize=8, colors='k')
            cbar.ax.set_xticklabels([f'{int(i):+d}' if i != 0 else '0' for i in ticks])
            cbar.ax.set_xlabel(r'\textbf{Geopotential Height Anomaly at 500 hPa [gpm]}', fontsize=12, labelpad=6)

            ### Subplot positioning
            cbar.ax.set_position([.1, .1, .8, .02])
            ax1.set_position([.03, .52, .45, .38])
            ax2.set_position([.52, .52, .45, .38])
            ax3.set_position([.03, .13, .45, .38])
            ax4.set_position([.52, .1635, .45, .38])

            ### Title
            fig.suptitle(rf'\textbf{{{season_name} North Atlantic Regimes}}', fontsize=18, y=.985)

            ### Annotation
            fig.text(.92, .95, rf'\textbf{{{first_year}–{end_year}}}', fontsize=11, ha='center')

            ### Save figure
            suffix = '_blackoutline' if style_name == 'black' else ''
            plt.savefig(os.path.join(fig_dir, f'Reanalysis_Z500_NA_Regimes_{season_name}_{first_year}–{end_year}{suffix}.png'), dpi=1000)
            plt.close()

        ### Progress
        print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')