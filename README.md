# A methodological break in the USDA Cropland Data Layer (2023→2024)

Code, configuration, derived tables and results for the article

> *A methodological break in the USDA Cropland Data Layer creates a discontinuity in mapped cropland and grassland change* (manuscript under review).

In 2024 the USDA NASS Cropland Data Layer (CDL) changed from a 30 m See5 classification (2008–2023) to a
10 m random-forest classification (2024 onwards), also released as a 30 m resample. This repository
quantifies how much of the land-cover change obtained by differencing consecutive CDLs across 2023→2024 is
attributable to that change of method, how it depends on field size and distance to field edges, how it
propagates to crop rotations, Crop Sequence Boundaries (CSB) and models trained on CDL labels, and how
far simple harmonization rules reduce it.

All processing is local (no Google Earth Engine): national CDL rasters are read in windows, and every
county, state and national metric is a ratio of sums of county transition counts.

## Repository layout

| Path | Content |
|---|---|
| `src/agrors/` | Python package with reusable functions (see below) |
| `scripts/` | Numbered scripts, run in order (see below) |
| `tests/` | `pytest` tests for parsing, joins, metrics and the error-matrix reader |
| `config/experiment.yaml` | All parameters: years, pilot states, CSB windows, edge buffers, HLS windows, bootstrap replicates, reference-sample design, source URLs, random seed (42) |
| `data/MANIFEST.csv` | Every raw file downloaded: source URL, download date, SHA-256, licence |
| `data/processed/transitions/` | County transition tables `{y0}_{y1}_{variant}.parquet` with columns `zone`, `from`, `to`, `n` (pixels) for every pair of consecutive years 2008–2025 and every product variant (below) |
| `data/processed/counties.csv` | County index used as `zone`: GEOID, state, name and area |
| `data/processed/quickstats_*.parquet` | NASS Quick Stats planted and harvested area by county and state, tidied |
| `data/interim/zones/counties_5070.gpkg` | County polygons (EPSG:5070) used for the county map |
| `results/tables/` | All result tables (CSV) and Markdown summaries |
| `results/figures/` | Figures 1–4 (vector PDF and 600 dpi PNG); `_pilot` versions use the six pilot states |
| `results/key_numbers.csv` | Every number cited in the article, with its source file and the row/column filter |
| `results/reference_sample/` | Stratified reference sample for photo-interpretation of 2023→2024 change |
| `results/environment.txt` | Exact package versions used (`pip freeze`) |
| `paper/literature_numbers.csv` | Figures taken from the literature and used in comparisons, with their source |
| `docs/REFERENCE_PROTOCOL.md` | Interpretation protocol for the reference sample (in Spanish) |

Product variants for the new-era years: `nass30` (30 m file published by NASS), `nn30` (true nearest
neighbour, the centre 10 m sub-pixel), `maj30` (3 × 3 majority of the 10 m file) and `native10` (10 m).
Land-cover groups and the crop definition follow the USDA CDL legend (`src/agrors/cdl.py`).

Code comments and docstrings are in Spanish.

## Installation

Python 3.11 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

Only `scripts/01b_quickstats.py` needs a credential: copy `.env.example` to `.env` and set
`NASS_API_KEY` (free key from https://quickstats.nass.usda.gov/api). `python scripts/check_env.py` checks
that the variables exist without printing them.

## What can be re-run from this repository alone

The derived tables in `data/processed/` are enough to recompute the national, state and county analyses
without downloading any raster:

```bash
python scripts/03b_e1_analysis.py full
python scripts/03g_nlcd_regimes.py
python scripts/03h_cropdef_sensitivity.py
python scripts/03i_crosswalk_sensitivity.py
python scripts/03j_counterfactuals.py
python scripts/03m_conversion_acres.py
python scripts/03f_e5_harmonization.py
python scripts/03l_edge_size.py
python scripts/04a_figures.py full
python scripts/04b_tables.py full
python scripts/04c_key_numbers.py
```

`03f` and `03l` start from `results/tables/e2_strata_pilot.csv`, the field-stratified counts written by
`03c`. `03k_accuracy_bound.py` also runs from this repository: it downloads the official NASS error
matrices (three zips, about 210 MB in total) on first use.

The remaining steps need the raw files, which are public but not redistributed here: the transition
tables themselves (02a–02c, 02i, 03a), the field-level analyses (03c, 03d, 03e), the provenance check
(02j), the Quick Stats uncertainty (03n), the data inventory (04d) and the reference sample (07a).
`scripts/01_download.py` and `scripts/01b_quickstats.py` retrieve them and record the SHA-256 of each
file in `data/MANIFEST.csv`; comparing it with the committed version shows whether NASS has revised a
file since our download. The full CONUS run needs about 150 GB of disk; each national 10 m zip is about
9–10 GB.

Scripts that take an argument accept `pilot` (six states: IA, IL, NE, KS, PA, CA) or `full` (CONUS). Download and
extraction steps skip files that are already present.

## Scripts

### 1. Downloads

| Script | What it does | Main output |
|---|---|---|
| `01_download.py` | Downloads the national CDL zips (30 m 2008–2025; 10 m 2024–2025), the national CSB files for the three windows, the Census 2023 county boundaries and CropSight-US. Resumable; records each file in the manifest. `[pilot\|full] [--skip-10m]` | `data/raw/`, `data/MANIFEST.csv` |
| `01b_quickstats.py` | County and state planted and harvested area for the main crops and hay, 2008–2025, from the Quick Stats API | `data/raw/quickstats/` |
| `01c_official_accuracy.py` | Reads the official NASS accuracy statistics from the CDL metadata (state reports to 2023, national from 2024) | `results/tables/official_accuracy*.csv` |
| `01d_cdl10m_pilot.py` | 10 m CDL 2024–2025 for the pilot states from the Source Cooperative Zarr mirror, which is pixel-identical to the NASS zips (checked by 02h) | `data/interim/cdl/` |

### 2. Preparation

| Script | What it does | Main output |
|---|---|---|
| `02a_extract_cdl.py` | Extracts the GeoTIFFs from the national CDL zips | `data/interim/cdl/` |
| `02b_county_raster.py` | Rasterizes the counties on the CDL 30 m grid (pixel-centre rule) | `county30.tif`, `counties.csv`, `counties_5070.gpkg` |
| `02c_majority.py` | 10 m → 30 m 3 × 3 majority (`maj30`) for 2024–2025, and agreement of the NASS 30 m file with the centre sub-pixel and with the majority, by state | `resampling_check.csv` |
| `02d_quickstats_tidy.py` | Tidies the raw Quick Stats files | `quickstats_{county,state}.parquet` |
| `02e_csb_extract.py` | Extracts CSB polygons for the pilot states to GeoParquet (EPSG:5070) | `data/interim/csb/` |
| `02f_hls_features.py` | Monthly (May–September) HLS v2 medians and CDL labels in the MGRS windows of the configuration, from Microsoft Planetary Computer | `data/processed/hls/` |
| `02g_cropsight_windows.py` | Prepares CropSight-US field objects 2021–2023 and proposes the HLS windows where they are densest | `e4_window_candidates.csv` |
| `02h_verify_10m.py` | Verifies the 10 m mirror and the NASS 30 m resample against the official 10 m file | `verify_10m.csv` |
| `02i_nn30.py` | True nearest-neighbour 10 m → 30 m for 2024 (`nn30`), because the NASS 2024 30 m file does not match the centre sub-pixel everywhere | `cdl_2024_nn30.tif` |
| `02j_provenance.py` | Raster profiles and checksums of the new-era files; agreement of the 2024 30 m file with each of the nine sub-pixel offsets and with the 3 × 3 majority in random blocks | `provenance_*.csv` |

### 3. Analyses

| Script | What it does | Main output |
|---|---|---|
| `03a_transitions.py` | County transition tables for every pair of consecutive years and variant | `data/processed/transitions/` |
| `03b_e1_analysis.py` | County, state and national change metrics (any class change, land-cover group change, cropland gross and net change, crop-type change, corn→corn), excess at the break relative to the adjacent pairs, step and persistence, county bootstrap, county-threshold sensitivity and comparison of CDL and Quick Stats areas | `e1_*.csv` |
| `03c_e2_strata.py` | Class change stratified by CSB field-size decile and distance to the field edge (pilot states; memory-intensive, checkpoints per state) | `e2_strata_pilot.csv`, `e2_size_deciles.csv` |
| `03d_e3_csb.py` | Consistency of CSB crop histories across windows that span the break: label change per polygon, corn–soybean alternation, and matching of CSB 2016–2023 and 2018–2025 polygons (IoU with bootstrap) | `e3_csb_*.csv` |
| `03e_e4_models.py` | Propagation to models trained on CDL labels: LightGBM on HLS medians, spatial block split, training on old-era versus new-era labels, scored against CDL and CropSight-US objects | `e4_metrics.csv` |
| `03f_e5_harmonization.py` | Harmonization rules (3 × 3 majority, 30 m and 60 m edge exclusion, combinations) and how much each reduces the excess | `e5_harmonization.csv` |
| `03g_nlcd_regimes.py` | Separates the method break from changes of the NLCD edition used for non-agricultural training, by classifying year pairs into regimes | `e1_pairs_regime_full.csv`, `e1_regime_summary_full.csv` |
| `03h_cropdef_sensitivity.py` | Results under three cropland definitions (USDA legend; without hay and forage classes; also without fallow), with bootstrap intervals | `e1_cropdef_*_full.csv` |
| `03i_crosswalk_sensitivity.py` | Frequency of the classes whose group assignment is debatable (58–60, 62 and 81) and sensitivity of the main figures to that assignment | `crosswalk_*.csv` |
| `03j_counterfactuals.py` | Alternative counterfactuals for the break and an additive decomposition separating the NLCD-switch effect, the new-era error level and the break itself | `e1_counterfactuals_full.csv` |
| `03k_accuracy_bound.py` | Downloads the official NASS state error matrices 2023–2025, aggregates them to land-cover groups and compares the group change expected if map errors were independent between years with the observed change | `accuracy_bound.csv`, `accuracy_forage_classes.csv` |
| `03l_edge_size.py` | Separates edge and field-size effects: edge × size cross table and weighted least squares with state and class fixed effects, with leave-one-state-out ranges | `e2_edge_size_cross.csv`, `e2_edge_size_model.csv` |
| `03m_conversion_acres.py` | Grassland↔cropland conversion acres that differencing consecutive CDLs would record, versus the 2014–2023 baseline | `conversion_acres_full.csv` |
| `03n_qs_cv.py` | Coverage and median of the published coefficients of variation of county planted area (corn, soybeans, cotton, 2022–2025) | `qs_cv_county.csv` |

### 4. Figures, tables and cited numbers

| Script | What it does | Main output |
|---|---|---|
| `04a_figures.py` | Figures 1–4: county map of the excess, national series 2008–2025, edge and size effects, rotations and CDL/Quick Stats area ratios | `results/figures/` |
| `04b_tables.py` | Markdown tables from the result CSVs | `results/tables/*.md` |
| `04c_key_numbers.py` | Every number cited in the article, each with its source file and filter | `results/key_numbers.csv` |
| `04d_data_inventory.py` | Inventory of the data used (counts read from the downloaded files) | `data_inventory.csv` |

### 5. Reference sample

| Script | What it does | Main output |
|---|---|---|
| `07a_reference_sample.py` | Two-stage stratified sample (strata = 2023→2024 group transition in the NASS 30 m product, rare change strata oversampled) for independent photo-interpretation, following `docs/REFERENCE_PROTOCOL.md` | `results/reference_sample/` |

`check_env.py` checks that `.env` has the required variables without printing their values.

## Library (`src/agrors/`)

| Module | Content |
|---|---|
| `cdl.py` | CDL class codes, land-cover groups following the USDA legend, block operations (3 × 3 majority, transition counts) |
| `grid.py` | National CDL grid (EPSG:5070) and windowed reading at 30 m and 10 m |
| `layers.py` | Windowed access to each product variant on the county grid |
| `metrics.py` | Change metrics from transition tables (ratios of sums), excess/step/persistence estimators and bootstrap |
| `fields.py` | Field strata from rasterized CSB polygons: distance to edge and size |
| `accuracy.py` | Reader for the official NASS error matrices (xlsx) and aggregation to groups |
| `quickstats.py` | Parsing of Quick Stats CSV files |
| `hls.py` | Monthly HLS v2 (L30 + S30) medians with Fmask cloud masking, from Planetary Computer |
| `io.py` | Resumable downloads and the download manifest (SHA-256 of every file) |
| `env.py` | Loading of credentials from `.env` |
| `plotting.py` | Figure style (journal column widths, palette, PDF/PNG export) |

## Data sources

USDA NASS Cropland Data Layer, Crop Sequence Boundaries and Quick Stats (public domain); US Census Bureau
cartographic boundary files (public domain); NASA LP DAAC Harmonized Landsat Sentinel-2 v2.0, accessed
through Microsoft Planetary Computer; CropSight-US v1.0.1 (CC BY 4.0). The raw files are not
redistributed; `data/MANIFEST.csv` lists their URLs and checksums.

The figures use the Source Sans 3 typeface when its TTF files are placed in `paper/fonts/`; otherwise
matplotlib falls back to DejaVu Sans.

## Licence

To be added.
