# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (2026-11-03, branch `dc2-optimisation`: altimetry scored as anomalies)

- **A model's sea level anomaly against the observed one** (`metrics/evaluator.py`): a source with
  `model_mdt` (`{path, endpoint_url, variable, offset}`, a zarr MDT local or on S3) scores the
  model's `zos - (MDT + offset)` against its anomaly (`sla_filtered`, `ssha_filtered`), edited by
  `max_abs_sla` and without the observation's MDT; the scores are labelled `sla`. Before, the
  observation was turned into an absolute height with its own MDT (CNES-CLS), ~47 cm above the
  datum of the models (GLO12's): the along-track rmse, mae and bias measured that offset (41 to
  52 cm for GLORYS, GloNet, GLO4ENS), only the crmse held. With GLO12's MDT and OceanBench's offset
  (-0.1148 m, >= 0.3), the biases are those of each model's datum (GloNet 0, GLORYS +6, GLO4ENS
  -5 cm) and the rmse 5-8 cm, the order of the Class 4 literature. Without `model_mdt`, unchanged.

### Added (2026-10-29, branch `dc2-optimisation`: probabilistic forecasts)

- **Ensemble forecasts** (`metrics/ensemble.py`): a forecast with a `member` dimension (or `ens`,
  `realization`, `number`) gets the deterministic scores on its ensemble mean, and `crps` (fair
  CRPS, Ferro 2014), `spread` and `ssr` (spread-skill ratio, Fortin et al. 2014) on its members,
  cos(latitude)-weighted. Against observations (`_run_class4_with_raw_per_bins`) every member is
  interpolated to the points, bins carry `crps`/`ens_var`/`n_members`, pooled by oceanbench's
  `pool_bins`; against GLORYS (`compute_metric`), at the gridded levels under the deterministic
  labels. The members stay lazy and are read one level at a time (50 members x 21 levels do not
  fit a worker). DC2's probabilistic baseline is Mercator's GLO4ENS (50 members).
- `config: local` sources (`get_dataset_from_config`): the source `dctools.submission` writes for
  a submitted model; it was missing from the port of the submission module, so `submit run` raised
  "Unknown dataset config name: local".

### Added (2026-10-29)

- **Energy by band of scales** (`metrics/spectra.py`), the probabilistic energy spectra of the
  DC2 proposal (WeatherBench 2): against a gridded reference, latitude rows cut into 64-point
  ocean segments, detrended, Hann-windowed, their variance summed per band of wavelengths
  (>500, 200-500, 100-200, <100 km): `energy_ratio_<band>` (members), `energy_ratio_mean_<band>`
  (ensemble mean), `err_ratio_<band>`. Any forecast (a deterministic one is a single member).
- Ensemble maps: the Class 4 bins carry `ssr` / `ssr_c`; against GLORYS, the bins of the gridded
  levels get `crps`, `crps_c`, `spread`, `ssr`, `ssr_c` (merged into the deterministic bins).

- `crps_c` and `ssr_c`: the ensemble scores once the bias (that of `crmse`, the ensemble mean's
  over the pooled points) is removed, for every variable; a second pass over the observation
  points in the Class 4 runner (~1.5x its time). What SSH reads (its reference-surface offset).

### Fixed (2026-10-29)

- **Leaderboard baselines**: they were looked for in `<results_directory>/../../<challenge>/
  leaderboard_results`, which only holds for the default output directory; elsewhere (a cluster
  scratch) the dcleaderboard package's demonstration results (noisy synthetic models) were shown
  as baselines. The challenge declares its root (`PROJECT_ROOT`, `leaderboard_root` in the YAML
  for a test run), and demonstration data are never used.
- The gridded scores' variables are named as the observation scores' ("Surface height" ->
  "Surface ssh", "50m eastward velocity" -> "50m u_current"): one name per variable in every table.

- **A NaN score dropped its whole batch**: `json.dump` wrote a NaN token, the consolidation's
  orjson rejected the file and skipped it as corrupted (all of a reference's results lost for a
  one-point level with a perfect ensemble mean). Batches are written with NaN as null, read back
  with the standard parser when orjson fails; undefined ratios are not emitted.

- **`submit run` evaluated nothing**: `run_from_config` imported its default evaluation class from
  `dc2.evaluation.evaluation`, which does not exist (`dc2.evaluation.dc2`); `ModelSubmission`
  takes the challenge's `evaluation_cls`. A submission's references are the challenge's
  `dataset_references` (GLORYS included), not its observation datasets only.
- The validator reports an ensemble's members ("Ensemble members"); an ensemble's `member`
  coordinate is no longer warned about as an unknown variable.

- **Forecasts as long as the horizon were never evaluated** (`build_forecast_index_from_catalog`):
  a file covered a lead day only when its `date_end` (its last daily stamp, 00:00) was at least
  a day after it, so the last lead day of every forecast was missing and a 10-day GloNet forecast
  with `n_days_forecast: 10` gave no forecast at all (also on `main`). A file now covers the days
  it holds.

- The submission validator required the variables a challenge derives (DC2's `sst`/`sss`, first
  level of `thetao`/`so`), failing every submission: the required ones are the source's
  `keep_variables`.
- `OCEANBENCH_VARIABLES` mapped `u_current` to the northward velocity and `v_current` to the
  eastward one (only a fallback for datasets without standard names: GLORYS and GloNet have them).
- Tests: `tests/test_ensemble_scores.py` (exact fair CRPS, independent of the ensemble size for a
  reliable ensemble, ssr ~1 reliable and < 1 narrow, Class 4 and gridded paths).

### Fixed (2026-10-28, branch `dc2-optimisation`: found by a real DC2 run)

A one-forecast DC2 run (GloNet 2024-01-03, two lead times, SWOT, Argo profiles and velocities,
GLORYS) on the EDITO data:

- **Evaluation options never reached `compute_metric`**: its connection parameters had no
  `eval_variables` (and the dataset's get standardized), so the SLA -> ADT conversion did not
  trigger in a real run. `get_dataset_from_config` attaches `source_eval_variables` (the YAML names)
  and `sla_to_adt`, `max_abs_sla`, `depth_range`. SWOT KaRIn: crmse 0.05 m.
- **Observation points outside the task window** (`preprocess_one_npoints` ignored `time_bounds`):
  the two-year Argo velocity file was scored whole at every lead time (207 183 points for a day;
  264 now), and a daily file over its whole day instead of +-`time_tolerance`.
- **Point observations at their own depth**: Argo parking-depth drift (and drogued drifters) were
  scored against the model's surface (the model was cut to its first level for any observation
  without a depth dimension). Grouped by depth, scored against the model interpolated to it,
  reported at `1000m` / `15m`. `depth_range` selects them (`[100, 3000]`: the surface drift of a
  float during its transmission is windage).
- `sla_filtered` (DUACS L3) is an `ssh` alias; a declared anomaly that is absent is an error.
- The per-bins output took only interval depth bins (it raised on levels); `w_sum` is not a map
  metric. A forecast longer than the horizon passes the conformance check.

### Fixed (2026-10-28, branch `dc2-optimisation`: DC2 scores against observations)

Measured on GloNet's 2024-01-03 forecast (lead 0) against the SARAL passes of that day: SSH RMSE
7.99 m (bias 3.46 m) before, 0.41 m (bias -0.31 m, centred 0.27 m) after.

- **SSH anomaly vs absolute height** (`metrics/evaluator.py`, `_anomaly_to_absolute_height`): the
  aliases map `ssha`, `sla`, `ssha_filtered`, `data_01__ku__ssha` and `zos` all to `ssh`, so an
  along-track anomaly was scored against a model's height above the geoid. The observation's MDT is
  now added back (ADT = SLA + MDT) before scoring, and the MDT is kept through the variable slimming.
  An anomaly without its MDT stops the evaluation (`sla_to_adt: false` for an absolute reference).
  Anomalies beyond `max_abs_sla` (2 m) are dropped: the DC2 along-track stores are unedited L2 (values
  of +-30 m over land and ice).
- **Class 4 scores pooled over bins** (`metrics/oceanbench_metrics.py`, with oceanbench's `pool_bins`):
  a variable's score was the unweighted mean of its bins' RMSE (a 1-point bin weighed as much as a full
  one; `count` was the mean count per bin). Scores are now those of all the points; bins carry `w_sum`;
  new `crmse` (bias removed), the SSH score to read.
- **Argo depth levels** (`metrics/oceanbench_metrics.py`): the levels of a profile were all binned
  together and reported as "Surface". Each level is now its own bin (`depth_bin`), matched to the
  model level by depth (not by index), and reported at the GLORYS12 levels.
- **Argo standard mode** (`dctools_core.argo_qc`, `data/connection/argo_data.py`): the loader dropped
  the QC flags and `DATA_MODE` (character variables) and read raw values. Now: adjusted values in A/D
  mode, QC flags 1-2 per variable, depth from pressure (TEOS-10; 21 m at 2000 dbar), in-situ ->
  potential temperature (0.16 degC at 2000 m), the shallowest value (<= 10 m) filling the levels above.
  The rules are those of oceanml3d-eval's GDAC reader.

### Added

- `dctools.submission` (`SubmissionValidator`, `ModelSubmission`), which DC2's `submit.py` imports and
  `main` did not have: ported from `performance_and_code_quality`, configs resolved by
  `resolve_dc_config` (a path, `$DC_CONFIG_DIR`, or `<cwd>/*/config`) instead of a path relative to
  the dctools install.
- `target_time_values: auto` (`data/coordinates.py`): lead times `0 .. n_days_forecast - 1`; an explicit
  list whose length differs from `n_days_forecast` is now an error.
- `gsw` dependency; CI for `dctools-core` (`.github/workflows/core-tests.yml`).
- Tests: `tests/test_dc2_scores.py`, `tests/test_submission_config.py`,
  `packages/dctools-core/tests/test_argo_qc.py` (synthetic: perfect forecast 0, offset = bias, noisier
  worse, points pooled not bins, profile levels apart).

### Changed

- **`dctools-core` extracted** (2026-09-23, `packages/dctools-core/`): the dependency-light data layer is
  now its own distribution so that `oceanml3d-eval` (no torch/dask.distributed/oceanbench) can reuse it.
  Moved, with re-exporting shims left at the old import paths: `dcio/loader.py` -> `dctools_core.loader`;
  alias tables + `get_standardized_var_name` -> `dctools_core.aliases`; `TIME_NAMES`, `get_time_bound_values`
  and the time-sanity helpers -> `dctools_core.timebounds`; `S3ConnectionConfig.create_fs` and
  `_is_transient_remote_error` -> `dctools_core.storage` (`make_filesystem`, `retry_remote`). New
  `dctools_core.altimetry` (granule-name period parsing, catalog-based listing). Behaviour changes: the
  moved modules log through stdlib `logging` (logger `dctools_core`) instead of loguru; an S3 config with no
  `url` now targets the EDITO gateway (`DCTOOLS_S3_ENDPOINT` to override) instead of AWS.

### Added

- **Documentation** (May 2026)
  - Complete README.md with installation, usage, and structure
  - Comprehensive installation guide with troubleshooting
  - Quick start guide with working code examples
  - Architecture documentation with design patterns and components
  - Configuration guide for YAML-based evaluation workflows
  - Contributing guidelines for developers
  - Full API reference for all modules
  - Updated main documentation index with clear navigation

- **Code Features**
  - Multi-source data loading from CMEMS, Argo, S3, and local files
  - Automatic coordinate system detection and normalization
  - Flexible data transformation pipeline
  - Distributed evaluation with Dask support
  - Metric computation (RMSE, MAE, bias, correlation)
  - Regional and depth-level aggregation
  - Configuration-driven evaluation workflows
  - Extensible architecture for custom implementations

### Changed

- **Documentation Structure**
  - Reorganized docs for better navigation
  - Separated usage guides, architecture, and configuration
  - Added comprehensive examples throughout documentation
  - Improved table of contents and cross-references

### Fixed

- **Documentation Completeness**
  - Filled in TODO sections in installation guide
  - Filled in TODO sections in quick start guide
  - Added missing data module to API reference
  - Updated project overview with current features

## [0.0.1] - 2025 (Initial Release)

### Added

- Initial Python package structure
- Core modules:
  - `dctools.data` - Data management and loading
  - `dctools.metrics` - Metric computation
  - `dctools.processing` - Data processing pipelines
  - `dctools.dcio` - Input/output operations
  - `dctools.utilities` - General utilities
  - `dctools.debug` - Debugging tools
- DC2 challenge implementation
- Support for multiple data sources (CMEMS, Argo, S3)
- Dask distributed computing integration
- xESMF-based interpolation
- Comprehensive test suite
- Poetry-based project management

### Project Structure

```
dc-tools/
├── dctools/          # Main package
├── dc2/              # DC2 challenge
├── scripts/          # Utility scripts
├── tests/            # Test suite
└── docs/             # Sphinx documentation
```

### Dependencies

- **Data**: xarray, netcdf4, zarr, h5py
- **Processing**: dask[distributed], xESMF, scipy
- **Sources**: argopy, copernicusmarine, s3fs
- **ML**: torch, torchvision, torchgeo
- **Geospatial**: cartopy, geopandas, shapely
- **Metrics**: oceanbench (fork)

## Roadmap

### Planned Features

- [ ] Additional data sources (ERA5, MODIS, SMOS)
- [ ] More evaluation metrics (spectral analysis, uncertainty)
- [ ] Performance optimization for multi-node clusters
- [ ] Graphical user interface for configuration
- [ ] Real-time evaluation dashboard
- [ ] Machine learning model integration

### Planned Improvements

- [ ] Increase test coverage to 80%+
- [ ] Add performance benchmarking suite
- [ ] Improve error messages and debugging
- [ ] Add support for additional domain-specific metrics
- [ ] Performance optimization for large datasets

## Version History

### Future Versions

- **v0.2.0**: Planned enhancements and bug fixes
- **v1.0.0**: Stable release with full feature set

### Current Version

- **v0.0.1**: Initial development release (May 2026 documentation update)

## Notes for Contributors

When adding changes, please:

1. Update this changelog under `[Unreleased]` section
2. Use categories: Added, Changed, Fixed, Deprecated, Removed
3. Include issue numbers when applicable
4. Keep entries brief and clear
5. Update version number when releasing

## Documentation Updates

The primary change in May 2026 was a comprehensive documentation update:

- **README.md**: Expanded from 50 to 300+ lines with full installation and usage guide
- **Installation Guide**: Added troubleshooting and multiple installation methods
- **Quick Start**: Created with working code examples
- **Architecture Guide**: Complete system design documentation
- **Configuration Guide**: YAML configuration reference
- **API Reference**: Updated with all modules and classes
- **Contributing Guide**: Added comprehensive contributor guidelines

All documentation now accurately reflects the current state of the codebase and can serve as the authoritative reference for users and developers.

---

**Last Updated**: May 3, 2026
