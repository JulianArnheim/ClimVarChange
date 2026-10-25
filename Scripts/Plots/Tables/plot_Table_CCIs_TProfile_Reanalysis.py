"""
Plot a table of climate change indices (CCIs) from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import matplotlib.pyplot as plt
import os
import time
import xarray as xr

def calculate_hist_mean(month, sector, region):
    var = f'ERA5_{month}_{sector}_{region}_CCIs'
    return f'{ERA5[var].sel(year=slice(epoch[0], epoch[1])).mean().item():.1f}°C'

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
data_dir = os.path.join(base_dir, 'Data/')
fig_dir = os.path.join(base_dir, 'Figures/Table/')

### Years and epochs
year_min, year_max = 1940, 2025
epochs = [(1981, 2010), (year_min, year_max)]

print(f'\n---- Climate Change Indices (CCIs) ----\n')

### Table parameters
months, regions, sectors = ['ANNUAL', 'DJF', 'JJA'], ['UTW', 'AA', 'PST', 'UPTG', 'LOTG'], ['NH (180°W–180°E)', 'AM (140°W–50°W)', 'AT (50°W–40°E)', 'AS (40°E–130°E)', 'PA (130°E–140°W)']

### Load data
ERA5 = xr.open_dataset(os.path.join(data_dir, f'ERA5_CCIs_{year_min}-{year_max}_Annual.nc'))

### Loop through epoch options
for epoch in epochs:
    epoch_text = f'{epoch[0]}–{epoch[1]}'

    ### Loop through month options
    for month in months:
        row_labels, col_labels = [rf'\textbf{{{s}}}' for s in sectors], [rf'\textbf{{{r}}}' for r in regions]

        ### Build table
        data_matrix = []
        for sector in [s.split()[0] for s in sectors]:
            data_matrix.append([calculate_hist_mean(month, sector, region) for region in regions])
        table_data = [[rf'\underline{{Period ({epoch_text})}}'] + col_labels]
        for label, row_vals in zip(row_labels, data_matrix):
            table_data.append([label] + row_vals)

        ### Figure design
        plt.rc('text', usetex=True)
        plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Avant Garde']})
        plt.rc('savefig', facecolor='w')
        plt.rc('xtick', color='k')
        plt.rc('ytick', color='k')
        plt.rc('axes', facecolor='w', labelcolor='k')

        ### Create figure
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.axis('off')
        table = ax.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[.27] + [.12] * 5)
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.25, 1.15)

        ### Title
        plt.title(rf'\textbf{{Climate Change Indices: {month} $\bar{{T}}$}}', fontsize=12, fontweight='bold', y=.91)

        ### Style borders and header cells
        for (row, col), cell in table.get_celld().items():
            if row == 0 or col == 0:
                cell.set_text_props(weight='bold', color='k')
                cell.set_facecolor('#e8e8e8')
            cell.set_edgecolor('k')
            if col == 0 and row > 0:
                cell._loc = 'right'
                cell.get_text().set_horizontalalignment('right')
            if (row > 1) or (row == 0 and col == 0):
                cell.get_text().set_fontsize(6)

        ### Save figure
        plt.subplots_adjust(bottom=.01)
        plt.savefig(os.path.join(fig_dir, f'Reanalysis_CCIs_{month}_{epoch_text}.png'), dpi=1000)
        plt.close(fig)

    ### Progress
    print(f'Check Mark [HH:MM:SS.SS]: {format_time(time.time() - start_time)}\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')