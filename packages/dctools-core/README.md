# dctools-core

The low-level layer of [`dctools`](../../README.md), split out so that it can be installed without the
data-challenge evaluator and its heavy dependencies (dask.distributed, torch, oceanbench, loguru).
`dctools` depends on it and re-exports its names from their historical locations, so nothing changes
for DC2; `oceanml3d-eval` uses it directly (`pip install "dctools-core[s3]"`).

| module | content | came from |
|---|---|---|
| `dctools_core.storage` | `make_filesystem()` (S3/MinIO endpoint + credential resolution, anonymous by default), `is_transient_remote_error()`, `retry_remote()` | `dctools.data.connection.config.S3ConnectionConfig.create_fs`, `connection_manager._is_transient_remote_error` |
| `dctools_core.loader` | `FileLoader.open_dataset_auto()` and its fixes for real-world zarr/NetCDF stores (nanosecond time units, stringified `scale_factor`, missing consolidated metadata, NetCDF3 magic-byte engine choice, multi-group files) | `dctools.dcio.loader` (moved) |
| `dctools_core.aliases` | `COORD_ALIASES`, `VARIABLES_ALIASES`, `get_standardized_var_name()` | `dctools.data.coordinates` (moved) |
| `dctools_core.timebounds` | `get_time_bound_values()` with the sanity window + ACDD `time_coverage_*` fallback | `dctools.data.connection.connection_manager` (moved) |
| `dctools_core.altimetry` | along-track granule filename parsing (`SRL_GPN_2PfP188_0004_20241209_232245_20241210_001302.CNES.zarr`, SWOT `..._20240111T232120_20240112T001246_...`), `list_granules()` filtered by date | DC2 `scripts/fix_altimetry_time_corruption.py` (regexes) |

Logging goes through the standard library (`logging.getLogger("dctools_core")`); `dctools` may route it
to loguru with an intercept handler.

```bash
pip install -e "packages/dctools-core[s3]"
pytest packages/dctools-core/tests
```
