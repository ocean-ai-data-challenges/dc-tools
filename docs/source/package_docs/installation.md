# Installation Guide

This guide provides detailed instructions for setting up dc-tools on your system.

## System Requirements

- **Python**: 3.11 or later (3.13 recommended)
- **Operating System**: Linux, macOS, or Windows (with WSL2 recommended)
- **Git**: For cloning the repository
- **Conda**: Highly recommended for managing complex dependencies

## Prerequisites

Before installing dc-tools, ensure you have:

1. **Conda installed**: [Install Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. **Git installed**: [Install Git](https://git-scm.com/)
3. **Poetry installed** (optional): Poetry will be installed via conda
4. **Sufficient disk space**: ~5GB for dependencies and test data

## Installation Methods

### Method 1: User Installation (Recommended)

For users who want to use dc-tools as a library:

```bash
# Clone the repository
git clone git@github.com:ocean-ai-data-challenges/dc-tools.git
cd dc-tools

# Create and activate conda environment
conda create -n dctools python=3.13 esmpy poetry -c conda-forge
conda activate dctools

# Install the package
poetry install
```

### Method 2: Developer Installation

For developers who want to contribute to dc-tools:

```bash
# Clone the repository
git clone git@github.com:ocean-ai-data-challenges/dc-tools.git
cd dc-tools

# Create and activate conda environment
conda create -n dctools-dev python=3.13 esmpy poetry -c conda-forge
conda activate dctools-dev

# Install with development dependencies
poetry install --with dev
```

This installs:
- Testing tools (`pytest`, `pytest-cov`)
- Code quality tools (`ruff`, `mypy`)
- Development utilities

### Method 3: Documentation Development

To modify the Sphinx documentation:

```bash
# Follow steps 1-3 from Method 2, then:
poetry install --with dev --with docs
```

This additionally installs:
- `sphinx` - Documentation generator
- `sphinx-rtd-theme` - ReadTheDocs theme
- `myst-parser` - Markdown support for Sphinx

## Important Notes

### xESMF Compatibility

There is a [known issue with `xESMF` installation](https://github.com/pangeo-data/xESMF/issues/269) that can cause import errors. If you encounter import errors:

1. **Reinstall the environment** (recommended):
   ```bash
   conda activate dctools
   conda remove -n dctools --all
   # Follow installation steps again
   ```

2. **Or try manual installation**:
   ```bash
   conda install -c conda-forge esmpy xesmf
   poetry install --no-cache
   ```

### CUDA Support (Optional)

For GPU acceleration with CuPy, install the appropriate CUDA version:

```bash
# For CUDA 11.x
conda install -c conda-forge cupy-cuda11x

# For CUDA 12.x (modify pyproject.toml first to use cupy-cuda12x)
```

## Verification

After installation, verify your setup:

```bash
# Activate environment
conda activate dctools

# Test basic import
python -c "import dctools; print(dctools.__version__)"

# Run quick tests
poetry run pytest -m unit --tb=short -q

# Check version and dependencies
poetry show
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'xesmf'`

**Solution**:
```bash
conda activate dctools
conda remove -n dctools --all
# Restart terminal, then reinstall from scratch
```

### Issue: `HDF5_USE_FILE_LOCKING` errors

These are automatically handled by dc-tools' environment initialization. If you see them:

```bash
export HDF5_USE_FILE_LOCKING=FALSE
export NETCDF4_DEACTIVATE_MPI=1
python your_script.py
```

### Issue: Memory errors during Dask operations

Configure Dask memory settings:
```python
import dask
dask.config.set({'dataframe.convert-string': False})
dask.config.set({'memory.target-fraction': 0.8})
```

### Issue: Permission denied on git clone

Use HTTPS instead of SSH:
```bash
git clone https://github.com/ocean-ai-data-challenges/dc-tools.git
```

## Building Documentation Locally

Once installed with docs dependencies:

```bash
cd docs
make html
# Open with:
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
start build\html\index.html  # Windows
```

## Environment Variables

dc-tools automatically configures these for compatibility, but you can override:

```bash
# Disable HDF5 file locking (required for Dask workers)
export HDF5_USE_FILE_LOCKING=FALSE

# Disable MPI for NetCDF (avoid conflicts with Dask)
export NETCDF4_DEACTIVATE_MPI=1

# Configure Argo cache
export ARGOPY_CACHEDIR=~/.cache/argopy
```

## Next Steps

After successful installation:

1. **[Quick Start Guide](quickstart.md)** - Learn basic usage
2. **[API Reference](api.md)** - Explore available functions and classes
3. **[Configuration Guide](../usage/config.md)** - Set up evaluation workflows
4. **[Data Challenges](../data_challenges/dc_index.md)** - Learn about specific challenges
