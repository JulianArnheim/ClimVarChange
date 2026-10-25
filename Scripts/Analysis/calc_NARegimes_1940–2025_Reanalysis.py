"""
Calculate 500 hPa geopotential height (Z500) anomaly clusters in the wintertime North Atlantic based on
Arnheim et al (2025), plus a persistence minimum (Falkena et al. 2020), from ERA5 reanalysis

Notes
-----
    Data : https://cds.climate.copernicus.eu/cdsapp#!/search?type=dataset
    Author : Julian B Arnheim
"""

### Import modules
import os
os.environ['OMP_NUM_THREADS'] = '11'  # set before numpy/sklearn load, to prevent the MKL memory-leak warning on Windows
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import time
import xarray as xr

def correct_clusters(labels, times, min_len=3):
    """Enforce a minimum run length while respecting temporal gaps."""
    labels, times = np.asarray(labels, dtype=int), np.asarray(times)
    if len(labels) == 0:
        return labels
    out = labels.copy()
    iteration = 0
    while True:
        iteration += 1
        print(f'Correcting clusters | Iteration {iteration}')
        changed, runs, start = False, [], 0
        ### Identify contiguous runs (breaking for label change or temporal discontinuity)
        while start < len(out):
            end = start + 1
            while (end < len(out) and out[end] == out[start] and (times[end] - times[end - 1]) / np.timedelta64(1, 'D') <= 1):
                end += 1
            runs.append((start, end, out[start]))
            start = end
        ### Correct short runs
        for i, (s, e, _) in enumerate(runs):
            if (e - s) >= min_len:
                continue
            left, right = runs[i - 1] if i > 0 else None, runs[i + 1] if i < len(runs) - 1 else None
            ### Ensure neighbors are temporally adjacent
            if left and (times[s] - times[left[1] - 1]) / np.timedelta64(1, 'D') > 1:
                left = None
            if right and (times[right[0]] - times[e - 1]) / np.timedelta64(1, 'D') > 1:
                right = None
            ### Choose replacement regime
            if left and right:
                fill_label = left[2] if (left[2] == right[2]) or (left[1] - left[0]) >= (right[1] - right[0]) else right[2]
            elif left:
                fill_label = left[2]
            elif right:
                fill_label = right[2]
            else:
                continue
            out[s:e] = fill_label
            changed = True
        ### Stop when fully converged
        if not changed:
            print(f'Cluster correction converged after {iteration} iterations\n')
            break
    return out

def find_nearest_point(cluster_data, coordinate):
    lat, lon = coordinate
    return cluster_data.sel(latitude=lat, longitude=lon, method='nearest')

def format_time(seconds):
    hours, minutes = int(seconds // 3600), int((seconds % 3600) // 60)
    return '{:02d}:{:02d}:{:.2f}'.format(hours, minutes, seconds % 60)

### Start time
start_time = time.time()

### Directories
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
data_dir = os.path.join(base_dir, 'Data')

### Years and epochs
year_min, year_max = 1940, 2025
epochs = [(year_min, year_max), (1981, 2010)]

print(f'\n---- Weather Regime Clusters ----\n')

### PCA (EOF analysis) specifications
cluster_order = ['NAO+', 'BL', 'NAO-', 'AR']
n_clusters = len(cluster_order)
n_EOFs, n_inits = 14, 500

### Loop through epochs
for start_year, end_year in epochs:
    print(f'Epoch {start_year}–{end_year}')

    ### Load data
    dataz_datasets = []
    for dec_start in range((start_year // 10) * 10, (end_year // 10) * 10 + 1, 10):
        dec_end = dec_start + 9
        if dec_end > year_max:
            dec_end = year_max
        filename = os.path.join(data_dir, f'ERA5_NH_{dec_start}-{dec_end}_Z500_Daily.nc')
        if os.path.exists(filename):
            ds_region = xr.open_dataset(filename, chunks={'time': 30})['z'].sel(latitude=slice(80, 20), longitude=slice(-90, 30))
            dataz_datasets.append(ds_region)
            print(f'Loaded: {dec_start}–{dec_end}')
        else:
            print(f'File not found: {dec_start}–{dec_end}')
    dataz_region = xr.concat(dataz_datasets, dim='time')
    print()

    ### Loop through seasons
    seasons = {'DJF': [12, 1, 2], 'DJFM': [12, 1, 2, 3]}
    winter_start, winter_end = start_year + 1 if start_year == 1940 else start_year, end_year
    for season_name, season_months in seasons.items():

        print(f'{season_name} | {winter_start}–{winter_end}')

        ### Seasonal selection
        if season_name == 'DJF':
            dataz_time = dataz_region.sel(time=slice(f'{winter_start - 1}-12-01', f'{winter_end}-02'))
        if season_name == 'DJFM':
            dataz_time = dataz_region.sel(time=slice(f'{winter_start - 1}-12-01', f'{winter_end}-03'))
        dataz_time = dataz_time.sel(time=~((dataz_time.time.dt.month == 12) & (dataz_time.time.dt.year == winter_end)))
        dataz_season = dataz_time.sel(time=dataz_time.time.dt.month.isin(season_months))

        ### Print details
        winter_year = xr.where(dataz_season.time.dt.month == 12, dataz_season.time.dt.year + 1, dataz_season.time.dt.year)
        dataz_season = dataz_season.sel(time=(winter_year >= winter_start) & (winter_year <= winter_end))
        winter_year = winter_year.sel(time=(winter_year >= winter_start) & (winter_year <= winter_end))
        n_seasons, n_days = len(np.unique(winter_year.values)), dataz_season.sizes['time']
        print(f'{n_seasons} winter seasons ({n_days} days)')

        ### Paths for saved files
        cluster_filename = os.path.join(data_dir, f'ERA5_NA_Z500_Clustered_Regimes_{season_name}_{winter_start}-{winter_end}.nc')
        cluster_ds_filenames = [os.path.join(data_dir, f'ERA5_NA_Cluster_{cluster}_{season_name}_{winter_start}-{winter_end}.nc') for cluster in cluster_order]

        ### Compute clusters if not available
        if not (os.path.exists(cluster_filename) and all(os.path.exists(f) for f in cluster_ds_filenames)):

            ### Smoothed daily climatology and detrended anomalies
            dataz_season = dataz_season.assign_coords(monthday=('time', (xr.where(dataz_season.time.dt.month == 12, '0-', '1-') + dataz_season.time.dt.strftime('%m-%d')).data))
            dataz_climo = dataz_season.groupby('monthday').mean(dim='time').compute().rolling(monthday=31, center=True, min_periods=1).mean()
            dataz_anoms = dataz_season.groupby('monthday') - dataz_season.groupby('monthday').mean(dim='time').compute().rolling(monthday=31, center=True, min_periods=1).mean()
            dataz_season_anoms = dataz_anoms.sel(time=np.isin(dataz_anoms['time.month'], season_months))
            dataz_season_anoms = dataz_season_anoms - xr.polyval(dataz_season_anoms['time'], dataz_season_anoms.polyfit(dim='time', deg=1).polyfit_coefficients)  # detrend each grid point
            print(f'Daily climatology, anomalies, and detrending complete')

            ### EOF analysis and k-means clustering
            weighted_dataz_season_anoms = dataz_season_anoms * np.sqrt(np.cos(np.deg2rad(dataz_season_anoms['latitude'])))  # cosine-latitude weighting (Chung and Nigam, 1999)
            anomalies_reshaped = weighted_dataz_season_anoms.stack(points=('latitude', 'longitude')).transpose('time', 'points')  # reshape for PCA
            EOF_scores = PCA(n_components=n_EOFs).fit_transform(anomalies_reshaped)
            labels = KMeans(n_clusters=n_clusters, random_state=0, n_init=n_inits).fit(EOF_scores).labels_  # standard Euclidean distance (L2 norm)
            print(f'Anomaly weighting, reshaping, EOF analysis, and k-means clustering complete')

            ### Extract points of interest once clusters are identified
            points_of_interest = {'NAO+': [(43, 7), (62, -28)], 'BL': [(65, 7), (58, -48)], 'NAO-': [(67, -57), (45, -15)], 'AR': [(54, -36), (65, 23)]}
            point_data = {}
            for category, (coord_h, coord_l) in points_of_interest.items():
                high, low = find_nearest_point(dataz_season_anoms, coord_h).compute(), find_nearest_point(dataz_season_anoms, coord_l).compute()
                point_data[category] = (high, low)

            ### Identify regime for each cluster
            cluster_titles = []
            for i in range(n_clusters):
                cluster_indices = np.where(labels == i)[0]
                diff_values = {category: high.isel(time=cluster_indices).mean(dim='time').item() - low.isel(time=cluster_indices).mean(dim='time').item()
                               for category, (high, low) in point_data.items()}
                cluster_titles.append(max(diff_values, key=diff_values.get))
            assert len(set(cluster_titles)) == n_clusters, f'Clusters could not be assigned to unique regimes: {cluster_titles}'
            print(f'Cluster identification complete\n')

            ### Cluster ordering
            label_mapping = {i: cluster_order.index(cluster_titles[i]) for i in range(n_clusters)}
            reordered_labels = correct_clusters(np.array([label_mapping[label] for label in labels]), dataz_season_anoms.time.values, min_len=3)
            cluster_frequencies = [(reordered_labels == i).mean() * 100 for i in range(n_clusters)]

            ### Save cluster data
            cluster_data = xr.Dataset({'cluster_frequencies': (('cluster'), cluster_frequencies), 'labels': (('time'), reordered_labels)},
                                      coords={'time': dataz_season_anoms.time, 'cluster': np.arange(n_clusters)},
                                      attrs={'description': 'Cluster labels and frequencies for Z500 NA regimes', 'cluster_order': ', '.join(cluster_order)})
            cluster_data.to_netcdf(cluster_filename)
            print(f'Saved {cluster_filename}, the cluster labels and frequencies')

            ### Precompute time indices per cluster to avoid repeated boolean indexing
            cluster_time_indices = {idx: np.where(reordered_labels == idx)[0] for idx in range(n_clusters)}
            cluster_climo_mean = dataz_climo.mean(dim='monthday').compute()

            ### Loop and compute means
            for idx, cluster_name in enumerate(cluster_order):
                cluster_anoms = dataz_season_anoms.isel(time=cluster_time_indices[idx]).mean(dim='time').compute()
                Z500_values = (cluster_anoms + cluster_climo_mean).assign_attrs(cluster_anoms.attrs)
                cluster_ds = xr.Dataset({'anomaly': cluster_anoms, 'Z500': Z500_values},
                    attrs={'description': 'Average anomalies and full geopotential height at 500 hPa', 'cluster': cluster_name, 'season': season_name, 'years': f'{winter_start}-{winter_end}'})
                cluster_ds.to_netcdf(cluster_ds_filenames[idx], encoding={var: {'zlib': True, 'complevel': 4} for var in cluster_ds.data_vars})
            print(f'Saved {", ".join(cluster_ds_filenames)}, the cluster mean anomalies\n')
        else:
            print(f'Cluster data and all cluster mean anomaly files already exist. Skipping computation.\n')

### End time
print(f'---- Script Duration [HH:MM:SS]: {format_time(time.time() - start_time)} ----\n')