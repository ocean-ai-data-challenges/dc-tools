# PPR Océan & Climat - Data Challenges Documentation

![Logo du PPR Océan & Climat](_static/Logo_PPR.jpg)

Welcome to the documentation for **dc-tools**, the framework for Ocean Data Challenges developed as part of the [PPR Océan & Climat](https://www.ocean-climat.fr/) initiative.

## Quick Navigation

### 📦 For Package Users

Get started with the dctools package:

- **[Installation Guide](package_docs/installation.md)** - Set up the package on your system
- **[Quick Start (5 min)](usage/quickstart.md)** - Learn basic usage patterns
- **[API Reference](package_docs/api.md)** - Complete API documentation

### 🎯 For Data Challenge Participants

Information about specific data challenges:

- **[Data Challenges Overview](data_challenges/dc_index.md)** - All available challenges
- **[DC1 - Sea Surface Height](data_challenges/DC1.md)** - SSH prediction challenge
- **[DC2 - Sea Surface Temperature](data_challenges/DC2.md)** - SST prediction challenge
- **[DC3 - Ocean Dynamics](data_challenges/DC3.md)** - Dynamics challenge
- **[DC4 - Chlorophyll](data_challenges/DC4.md)** - Biogeochemistry challenge
- **[DC5 - Advanced Challenge](data_challenges/DC5.md)** - Advanced topics

### 👥 For Developers

Extend and contribute to the framework:

- **[Package Documentation](package_docs/dctools_index.md)** - Deep dive into dctools
- **[Developer Setup](package_docs/installation.md#method-2-developer-installation)** - Set up development environment
- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute (if available)

### ⚙️ For System Administrators

Deploy and manage data challenges:

- **[Installation for Production](package_docs/installation.md)** - Production deployment
- **[Dask Configuration](usage/quickstart.md#6-distributed-evaluation-with-dask)** - Cluster setup
- **[Dependencies](../pyproject.toml)** - Complete dependency list

## What is dc-tools?

dc-tools is a flexible Python framework for:

1. **Loading data** from multiple sources (CMEMS, Argo, S3, local files)
2. **Processing data** with coordinate normalization and interpolation
3. **Computing metrics** (RMSE, MAE, bias, class4 validation)
4. **Evaluating models** at scale with Dask parallelization
5. **Managing workflows** through YAML configuration files

## Key Features

```{admonition} Multi-Source Data Loading
Load data from CMEMS, Argo, S3, and local files with automatic source detection.
```

```{admonition} Flexible Coordinates
Automatically detect and normalize various coordinate naming conventions (lat/latitude/nav_lat, etc).
```

```{admonition} Distributed Evaluation
Built-in Dask support for evaluating on clusters with automatic memory management.
```

```{admonition} Configuration-Driven
Define evaluation workflows in YAML for reproducibility and version control.
```

## Typical Workflow

```
1. Install dctools
        ↓
2. Configure data sources (YAML or code)
        ↓
3. Load and normalize data
        ↓
4. Interpolate to evaluation grid
        ↓
5. Compute metrics (locally or on cluster)
        ↓
6. Save and analyze results
```

## Recent Changes

This documentation has been updated to reflect the current state of the codebase (May 2026):

- ✅ Complete installation instructions with troubleshooting
- ✅ Comprehensive quick start guide with working code examples
- ✅ Full API reference for all modules
- ✅ Architecture documentation
- ✅ Common use cases and patterns
- ✅ Configuration file examples

## Getting Help

- **[Quick Start Guide](usage/quickstart.md)** - Start here for basic usage
- **[API Reference](package_docs/api.md)** - Look up specific functions and classes
- **[GitHub Issues](https://github.com/ppr-ocean-ia/dc-tools/issues)** - Report bugs or ask questions
- **[GitHub Discussions](https://github.com/ppr-ocean-ia/dc-tools/discussions)** - General questions and discussions
- **[Main Repository](https://github.com/ppr-ocean-ia/dc-tools)** - Source code and development

## Project Information

- **Framework**: Python 3.11+
- **License**: GPL-3.0
- **Authors**: Kamel Ait Mohand, Guillermo Cossio
- **Organization**: [PPR Océan & Climat](https://www.ocean-climat.fr/)
- **Repository**: [github.com/ppr-ocean-ia/dc-tools](https://github.com/ppr-ocean-ia/dc-tools)

## Main Dependencies

- **Data**: xarray, netcdf4, zarr, h5py
- **Processing**: dask[distributed], xESMF, scipy
- **Sources**: argopy, copernicusmarine, s3fs
- **ML**: torch, torchvision, torchgeo
- **Metrics**: Custom OceanBench fork

See [pyproject.toml](../pyproject.toml) for the complete dependency list.

---

```{toctree}
:maxdepth: 2
:caption: Documentation
:hidden:

package_docs/dctools_index.md
usage/quickstart.md
package_docs/installation.md
package_docs/api.md
data_challenges/dc_index.md
```

**Last updated**: May 2026 | **Status**: Complete and current
