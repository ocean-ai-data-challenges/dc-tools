# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
