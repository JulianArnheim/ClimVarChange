# ClimVarChange

Scripts for computing and visualizing diagnostics of climate variability and change, with a focus
on Arctic amplification, midlatitude jet stream and blocking behavior, and their linkages, built on
reanalysis data.

## Overview

The codebase follows a four-stage pipeline, reflected directly in the `Scripts/` subfolders:

1. **Download** (`Scripts/Utilities/download/`) pulls raw data from its original source (CDS API,
   NOAA PSL and CPC, PIOMAS, etc.) into `Data/raw/`. The ERSSTv6, HadISST and NOAA NAO scripts produce
   analysis-ready files and write them straight to `Data/`, so they have no preprocess step.
2. **Preprocess** (`Scripts/Utilities/preprocess/`) regrids, subsets, and cleans the raw data into
   analysis-ready NetCDF files in `Data/`.
3. **Analysis** (`Scripts/Analysis/`) computes derived diagnostics from the preprocessed data, such
   as Arctic amplification indices, blocking frequency, jet latitude/speed indices, and North
   Atlantic circulation regimes, saving the results back to `Data/` as NetCDF.
4. **Plots** (`Scripts/Plots/`) reads from `Data/` and produces the figures in `Figures/`.

Each script is self-contained and meant to be run directly, top to bottom.

## Directory structure

```
ClimVarChange/
├── Data/                         Raw and processed NetCDF data
│   └── raw/                      Untouched downloads, before preprocessing
├── Figures/                      Output figures
├── Scripts/
│   ├── Analysis/                 Derived climate diagnostics
│   │   calc_AA_T2M, calc_BLO_Sectors_5d, calc_CCIs, calc_JVI_Sectors,
│   │   calc_NA_JLI_JSI, calc_NARegimes, calc_SIN_Sectors, calc_ZON_Sectors
│   ├── Utilities/                setup_ClimVarChange_Folders (creates the Data/ and Figures/ layout)
│   │   ├── download/             Raw data retrieval (CDS API/ERA5, ERSSTv6,
│   │   │                         HadISST, NOAA PSL daily NAO and monthly ONI,
│   │   │                         NOAA CPC monthly NAO/EA and RONI, PIOMAS)
│   │   └── preprocess/           Regridding/cleaning of raw data
│   └── Plots/
│       ├── Bars/                 Change in the meridional temperature difference by level
│       ├── Histograms/
│       │   ├── JLI/              Jet Latitude/Speed Index histograms
│       │   └── SIN/              Sinuosity Index histograms
│       ├── Profiles/             Vertical temperature/wind profile figures
│       │   ├── Amplification/
│       │   ├── Diffs/{T,TU,TV}/
│       │   ├── Hist/{T,TPsi,TU,TV}/
│       │   ├── Hist Std/
│       │   └── Slopes/{T,TU,TV}/ (T/ also has a Series/ subfolder)
│       ├── Scatter/
│       │   ├── AT/
│       │   └── NH/
│       ├── Spatial/              Map figures
│       │   ├── Baroclinicity/, DDs/, ICEFRAC/Arctic/, NARegimes/, PRECT/Global/,
│       │   │   SIT/, SLP/, SST/{Global,NAWH}/, T2M/{Arctic,CONUS,Global,NH}/, UV/Global/
│       │   └── plot_U850_MapHistNA_NAO, plot_Z500_MapStormtrackNA
│       ├── Tables/
│       ├── Timeseries/           plot_CCIs, plot_Z500_NARegimes at the folder root
│       │   ├── DDs/{BLO,JVI,SIN,ZON}/
│       │   ├── ENSO/             ONI and RONI with El Niño/La Niña strength categories
│       │   ├── ICEFRAC/{SIA,SIE,SIV}/
│       │   ├── PRECT/{Arctic,Global,NH}/
│       │   ├── SST/{Arctic,Global,NAWH}/
│       │   ├── T2M/{Arctic,Global,NH}/
│       │   └── Z500/
│       └── Zonal Mean/
├── bin/                          build_image.sh and run_container.sh (Docker helpers)
├── Dockerfile                    Container image with the conda environment and TeX
├── .dockerignore
├── requirements.txt
├── LICENSE
├── README.md
├── SCRIPTRULES.txt               Script conventions
└── .gitignore
```

## Data and figure folders

The folder layout below is part of the code base. Run `Scripts/Utilities/setup_ClimVarChange_Folders.py`
once before anything else. It creates every folder listed here and leaves existing folders untouched.
No other script creates folders, so a missing folder makes `xr.Dataset.to_netcdf`, the download calls
or `plt.savefig` fail with a `FileNotFoundError`. Download scripts write to `Data/raw/`
(or straight to `Data/` for ERSSTv6, HadISST and the NOAA NAO index), and every other script reads
from and writes to `Data/`. Each plot script sets its output folder in `fig_dir`.

Figure folders do not always mirror the script folders, because `Figures/` is organised by region
first and then by plot type. Timeseries, histogram and zonal-mean figures follow the script layout
(`Scripts/Plots/Timeseries/T2M/Arctic/` writes to `Figures/Timeseries/T2M/Arctic/`). Map figures go
to `Figures/Spatial/<Region>/<Variable>/<Diff|Historical|Historical Std|Slope>/` (for example
`Scripts/Plots/Spatial/T2M/Global/` writes to `Figures/Spatial/Global/T2M/Slope/`). Profile figures
go to `Figures/Profile/<Type>/<Global|NH>/<10 top|100 top>/`, and profile slope figures add a season
subfolder. The Series profile slopes and the CONUS trend maps also save a copy without stippling to a
`no_stippling/` subfolder, with `_no_stippling` appended to the file name. Braces list sibling folders.

```
Data/
└── raw/
Figures/
├── Bars/
├── Histograms/{JLI,SIN}/
├── Profile/
│   ├── Amplification/{Global,NH}/{10 top,100 top}/
│   ├── Diffs/{Global,NH}/{10 top,100 top}/{T,TU,TV}/
│   ├── Historical/{Global,NH}/{10 top,100 top}/{T,TPsi,TU,TV}/
│   ├── Historical Std/{Global,NH}/{10 top,100 top}/
│   └── Slopes/
│       ├── {Global,NH}/{10 top,100 top}/{T,TU,TV}/{ANNUAL,DJF,JJA}/
│       └── Series/no_stippling/
├── Scatter/{AT,NH}/
├── Spatial/
│   ├── Arctic/
│   │   ├── {ICEFRAC,SIT}/{Diff,Historical,Historical Std,Slope}/
│   │   └── T2M/Local AA/{Latitudes,Periods,Sweeney2023}/
│   ├── CONUS/T2M/Slope/no_stippling/
│   ├── Global/
│   │   ├── {PRECT,SLP,SST,T2M}/{Diff,Historical,Historical Std,Slope}/
│   │   └── {UV10,UV100,UV250,UV500,UV700,UV850}/Historical/
│   ├── NA/
│   │   ├── {Baroclinicity,NAO,Regimes,Stormtrack}/
│   │   └── SST/NAWH/{ERA5,ERSSTv6,HadISST}/
│   └── NH/
│       ├── {BLO,SIN}/
│       └── T2M/{Diff,Historical,Historical Std,Slope}/
├── Table/
├── Timeseries/
│   ├── CCIs/{letters,no letters}/anom/
│   ├── DDs/{BLO,JVI,SIN,ZON}/
│   ├── ENSO/
│   ├── ICEFRAC/{SIA,SIE,SIV}/
│   ├── NAO/gifs/
│   ├── {PRECT,T2M}/{Arctic,Global,NH}/
│   ├── SST/{Arctic,Global,NAWH}/
│   └── Z500/
└── Zonal Mean/{SST,T2M}/
```

Within a leaf folder, most scripts exist in small families that share a common base and differ by a
handful of parameters, most often the vertical extent (`10top` vs. `100top`), hemisphere (`NH` vs.
global/full-latitude), or region (`Global`/`NH`/`Arctic`/`CONUS`).

## Naming conventions

**Scripts** (`Scripts/**/*.py`): every name starts with one of `calc_`, `download_`, `preprocess_`
or `plot_`. The one exception is the folder setup script, `setup_ClimVarChange_Folders.py`. Download and preprocess scripts follow
`<verb>_<Source/Dataset>_<Domain>_<Variable>_<Cadence>.py` (e.g. `preprocess_PIOMAS_Arctic_SIC_Monthly.py`).
Analysis and plot scripts name the diagnostic or variable, the region or variant and the data type,
ending in `_Reanalysis` (e.g. `calc_SIN_Sectors_1940–2025_Reanalysis.py`,
`plot_Arctic_T2M_Anoms_Reanalysis.py`, `plot_TProfileAmpNH_100top_Reanalysis.py`). Year ranges in
filenames use a Unicode en dash (`–`, U+2013), e.g. `plot_AA_T2M_1940–2025_Reanalysis.py`, never an
ASCII hyphen; hyphens inside a variable name (e.g. `U925-U700`) are ordinary ASCII hyphens.

**Data files** (`Data/**/*.nc`): `<Source>_<Domain>_<StartYear>-<EndYear>_<Variable>_<Cadence>[_raw].nc`,
with an ordinary ASCII hyphen in the year range (not the en dash used in script and figure names).
Raw, untouched downloads carry a `_raw` suffix and live under `Data/raw/`; preprocessed and derived
files drop the suffix and live directly under `Data/`. Raw PIOMAS files keep their source names and
formats (`area.H<year>` and `heff.H<year>` binaries and the `grid.dat.txt` model grid). Examples:
- `Data/raw/ERA5_NH_1940-1949_Z500_Daily_raw.nc` → preprocessed to `Data/ERA5_NH_1940-1949_Z500_Daily.nc`
- `Data/ERA5_Global_1940-2025_SST_Monthly.nc` (a full-period preprocessed file)
- `Data/ERA5_CCIs_1940-2025_Annual.nc` (an Analysis-derived diagnostic)

Analysis outputs follow `<Source>[_<Domain>]_<Diagnostic>[_<Variant>]_<Start>-<End>_<Cadence>.nc`
(e.g. `ERA5_BLO_1d_1940-1949_Daily.nc`). Plot scripts may cache intermediate products as
`<Source>_<Domain>_<Start>-<End>_<Variable>[_Slopes|_PValues]_<Season>_<Sector>[_AnnualZonalMeans].nc`
(e.g. `ERA5_Global_1940-2025_TProfile_DJF_NH_AnnualZonalMeans.nc`). Other caches are annual means
(`ERA5_Global_1940-2025_<Variable>_Annual.nc`, `<Dataset>_NAWH_SST[_Index]_<Season>_1940-2025_Annual.nc`),
epoch amplification profiles (`ERA5_Global_<Start>-<End>_to_<Start>-<End>_TProfileAmp_<Season>_<Sector>.nc`)
and the WACE trends (`ERA5_Global_1998-2012_{T2M,SLP}_{Slopes,PValues}_DJF.nc`). Delete a cache to
force it to be recomputed after the upstream data or the method change.

A few data files do not follow the pattern above. The NOAA index files have no domain
(`CPC_NAO_EA_1950-2025_Monthly.nc`, `PSL_NAO_1948-2025_Daily.nc`). The raw ONI and RONI downloads
are text tables and the raw PIOMAS volume file is a CSV-style text table. The regime outputs end in
`<Season>_<FirstWinter>-<LastWinter>.nc`.

Multi-decade daily datasets (ERA5 daily Z500 and NA winds) and the monthly T, U and V profiles are
chunked into 10-year files by the download/preprocess scripts rather than one full-period file.

**Figures** (`Figures/**/*.png`): `Reanalysis_<Region>_<Variable>[_Anoms|_Change|_Diff|_Std][_Months/Seasons/...]_
<StartYear>–<EndYear>[_extra].png`, with the Unicode en dash in the year range (matching script
names, not Data file names). Examples:
- `Reanalysis_Global_T2M_1940–2025.png` (seasonal-cycle-by-year plot, monthly cadence)
- `Reanalysis_Global_T2M_Anoms_Seasons_1940–2025.png` (annual-anomaly trend, by season)
- `Reanalysis_AA_T2M_1940–2025_66.5N.png` (Arctic amplification, region defined by latitude bound)
- `Reanalysis_Arctic_ICEFRAC_Diff_DJF_1941–1970_1991–2020.png` (difference between two epochs)

Figures from non-reanalysis products replace the `Reanalysis_` prefix with the dataset name
(`PIOMAS_Arctic_SIT_...`, `ERSSTv6_NA_SST_Change_...`). Known exceptions to the pattern:
- Profile figures put the years first (`Reanalysis_1981–2010_TProfile_100top_DJF.png`,
  `Reanalysis_1940–2025_TProfileSlope_100top_DJF_NH.png`).
- The PIOMAS SIT climatology maps are `Reanalysis_SIT_<Season>.png` and `Reanalysis_SIT_Std_<Season>.png`,
  with no years and no `PIOMAS_` prefix.
- The PIOMAS SIV timeseries keep the `Reanalysis_` prefix (`Reanalysis_PIOMAS_SIV_...`).
- The NAWH trend map names its ERA5 figures `ERA5_NA_SST_Change_...`, like the other two datasets.

## Data sources

Currently used by the download/preprocess scripts:

- **ERA5** (Climate Data Store API): 2-m temperature, sea surface temperature, sea level pressure,
  geopotential height, precipitation, sea ice concentration, 10/100-m and profile winds, temperature
  profile. Most variables are global monthly fields; North Atlantic 925-700 hPa winds and Northern
  Hemisphere 500 hPa height are also retrieved as regional daily fields.
- **ERSSTv6** and **HadISST**: sea surface temperature.
- **NOAA PSL**: daily NAO index, computed from NCEP/NCAR Reanalysis 1 500 hPa heights (used by the
  Histograms/JLI NAO scripts and the Spatial U850 NAO map), and the monthly Oceanic Niño Index
  (ONI) from ERSSTv6 Niño 3.4 SST anomalies.
- **NOAA CPC**: monthly NAO and EA teleconnection indices (used by the NAO/EA scatter and
  timeseries scripts), and the monthly Relative Oceanic Niño Index (RONI, from ERSSTv5), which CPC
  adopted as its primary ENSO index in 2026.
- **PIOMAS**: sea ice thickness, volume and concentration from a coupled ice-ocean model with data
  assimilation (Zhang & Rothrock 2003), not an observational dataset. Thickness is the grid-cell
  mean (effective) thickness.

ERA5 sea-ice concentration before 1979 is prescribed from reconstructions rather than observed by
satellite. The ICEFRAC trend maps loop over start years from 1940 to 1980, and the ICEFRAC
epoch-difference maps compare 1941–1970 with 1991–2020, so any trend or difference that includes
years before 1979 partly reflects those reconstructions.

## Conventions

**Sectors.** Sector keys are kept short in code, titles, and variable and cache names. As a sector key, `NH`
means all longitudes (180°W–180°E). In profile script and folder names (`plot_TProfileChangeNH_...`,
`Figures/Profile/.../NH/`), `NH` instead means 0–90°N rather than the full 90°S–90°N section.

| Key | Longitudes |
|---|---|
| NH | 180°W–180°E (all longitudes) |
| AM | 140°W–50°W |
| AT | 50°W–40°E |
| AS | 40°E–130°E |
| PA | 130°E–140°W |

**Arctic boundary.** The regional Arctic timeseries (T2M, SST, PRECT) use 66.5°N. The `plot_AA_*`
timeseries loop over 60, 65, 66.5 and 70°N. The local-amplification maps and `plot_LocalAmpArctic`
use 70°N, and `plot_LocalAmpArctic_Latitudes` loops over 60 to 85°N (including 66.56°N). `calc_AA_T2M`
uses 70°N. `calc_CCIs` uses 60°N for its Arctic amplification index and 65°N for the temperature
gradients. Each figure states its boundary.

**DJF.** Winters are labelled by the January year. Because the records start in January, the first
DJF is `year_min + 1`.

**Significance.** Trend p-values come from a two-sided t-test on the least-squares slope, with the
degrees of freedom reduced for lag-1 autocorrelation of the residuals (effective sample size, Santer
et al. 2000). Trend stippling and masks then use an area-weighted Benjamini-Hochberg false discovery
rate (FDR) at α_FDR = 0.10 (`sig_threshold = 2 * .05`). Maps weight by cos(latitude). Profiles
weight by cos(latitude) times pressure thickness and include only the plotted domain (up to 100 or
10 hPa, and 0–90°N for NH profiles). Following Wilks (2016), α_FDR = 2α_global controls field
significance at the 95% confidence level (α_global = 0.05) for spatially correlated fields, so
figures describe shaded or stippled trends as significant at the 95% confidence level after
area-weighted (or area- and pressure-weighted) FDR correction. The one exception,
`plot_T2M_MapChangeGlobal_WACE_Reanalysis.py`, stipples local p < 0.05 without FDR correction.

**Area means.** Area means use cos(latitude) weights over valid points only. Sea surface temperature
means are two-dimensional, so each latitude band counts in proportion to its ocean area.

**Daily fields and diagnostics.** "Daily" ERA5 fields are single 12 UTC snapshots, not daily means.
JLI and JSI are computed from 12 UTC zonal-mean zonal wind over the North Atlantic at 700–925 hPa,
and the JLI histograms use 850 hPa unless they compare levels. The blocking index applies the
meridional Z500 gradient-reversal criterion at each longitude with a five-day persistence
requirement and no spatial-extent criterion. The JVI is computed once per calendar month and
repeated on each day of that month in the daily files. The JVI scatter plots pair each daily SIN or
ZON value with the JVI of its month, so the colour bar counts days. Sinuosity is the great-circle
length of the Z500 isohypse at the 30–70°N mean height divided by the length of the 50°N circle
across the sector (Cattiaux et al. 2016). Because the reference is fixed at 50°N, a flat isohypse
lying poleward of 50°N is shorter than the reference, so sector SIN can fall slightly below 1. Over
1940–2025, values below 1 occur on 0.25% of days in AT (minimum 0.94, mostly in late summer and
autumn), 0.03% in AS, 0.01% in AM and 0.01% in PA. The NH index stays above 1. The North Atlantic
regimes are clustered from linearly detrended daily Z500 anomalies.

## Environment

`requirements.txt` lists the packages actually imported across `Scripts/`: `numpy`, `xarray`,
`netCDF4` (the NetCDF backend xarray reads/writes through), `pandas`, `matplotlib`, `cartopy`,
`cmocean`, `scipy` (`scipy.stats.linregress` and `scipy.stats.t` for trends,
`scipy.interpolate.griddata` for regridding), `statsmodels`, `scikit-learn`, `scikit-image`, `shapely`, `dask`, `cdsapi` and
`requests` (download scripts), and `paramiko` (the PIOMAS SFTP download). Install with either
`conda install --file requirements.txt` or `pip install -r requirements.txt`.

Figures are rendered with LaTeX text (`matplotlib.rc('text', usetex=True)`), which requires a
working TeX installation (`dvipng` and `texlive`) separate from the Python packages above.

**Paths.** Every script resolves the project root relative to its own location (`__file__`), so the
code runs from any folder name and on any machine without edits.

**Docker.** `bin/build_image.sh` builds the `climvarchange` image from `Dockerfile` (conda environment
plus TeX), and `bin/run_container.sh <script>` runs a script with the project mounted at
`/usr/local/src/ClimVarChange`. Run it from the project root, since it mounts the current folder,
e.g. `bin/run_container.sh Scripts/Utilities/setup_ClimVarChange_Folders.py`.

## Acknowledgments

The data visualization approach used throughout this repository, particularly the plotting style
and figure conventions, is heavily inspired by the code and figures of Zachary Labe (https://zacklabe.com/).
Many thanks to him for making his work publicly accessible.


## Author

Julian B. Arnheim, Department of Earth System Science, University of California, Irvine
(jarnheim@uci.edu)