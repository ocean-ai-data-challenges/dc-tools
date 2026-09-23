# Contributing to dc-tools

Thank you for your interest in contributing to dc-tools! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please treat all individuals with respect and courtesy.

## Ways to Contribute

### 1. Report Bugs

If you find a bug, please open an issue on GitHub with:

- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Expected behavior** vs. actual behavior
- **Environment details** (Python version, OS, dependencies)
- **Error messages** and tracebacks (if applicable)

**Example**:
```
Title: "Import error with xESMF on M1 Mac"

Description:
When running `poetry install`, I get an error with xESMF.

Steps:
1. Create conda environment: `conda create -n dctools python=3.13 esmpy poetry`
2. Install: `poetry install --with dev`

Error:
```

### 2. Suggest Enhancements

Have an idea for improvement? Open an issue with:

- **Use case**: What problem does this solve?
- **Proposed solution**: How should it work?
- **Alternative approaches**: Other options considered?

### 3. Fix Bugs

Pick an issue marked `good-first-issue` or `bug` and submit a PR (see below).

### 4. Add Features

Implement a new feature:

- Check for existing issues/discussions first
- Start an issue to discuss the approach
- Get feedback before implementing
- Submit a PR once complete

### 5. Improve Documentation

- Fix typos or unclear explanations
- Add missing examples
- Improve code comments
- Add docstrings to undocumented functions
- Update documentation for recently changed features

### 6. Improve Tests

- Add tests for untested code
- Expand integration tests
- Add performance benchmarks
- Fix flaky tests

## Development Workflow

### Step 1: Set Up Development Environment

```bash
# Clone repository
git clone git@github.com:ppr-ocean-ia/dc-tools.git
cd dc-tools

# Create conda environment
conda create -n dctools-dev python=3.13 esmpy poetry -c conda-forge
conda activate dctools-dev

# Install with all dependencies
poetry install --with dev --with docs

# Verify setup
poetry run poe all
```

### Step 2: Create a Feature Branch

```bash
# Update main branch
git checkout main
git pull origin main

# Create feature branch (use descriptive name)
git checkout -b feature/my-feature  # For new features
git checkout -b fix/issue-number     # For bug fixes
git checkout -b doc/improve-readme   # For documentation
```

### Step 3: Make Your Changes

**Code guidelines**:

1. **Follow PEP 8** - Code style enforced by `ruff`
2. **Add type hints** - Include Python type annotations
3. **Write docstrings** - Use NumPy style docstrings:

```python
def my_function(data: xr.Dataset, region: str) -> Dict[str, float]:
    """
    Compute metrics for a specific region.
    
    Parameters
    ----------
    data : xr.Dataset
        Input dataset with variables to evaluate
    region : str
        Name of region (e.g., 'global', 'tropical')
    
    Returns
    -------
    Dict[str, float]
        Dictionary mapping variable names to RMSE values
    
    Examples
    --------
    >>> results = my_function(dataset, 'tropical')
    >>> print(results['THETAO'])
    0.5
    """
    pass
```

4. **Add tests** - Include tests for new functionality

```python
# tests/test_my_feature.py
def test_my_function_basic():
    """Test basic functionality"""
    result = my_function(sample_data, 'global')
    assert isinstance(result, dict)
    assert 'THETAO' in result

def test_my_function_tropical_region():
    """Test tropical region specifically"""
    result = my_function(sample_data, 'tropical')
    assert result['THETAO'] < 0.8  # Expected range
```

5. **Update documentation** - Add docstrings, update guides if needed

### Step 4: Test Your Changes

```bash
# Run fast tests only (< 1s each)
poetry run poe test-fast

# Run all tests (unit + integration)
poetry run poe test-coverage

# Run linting and type checking
poetry run poe lint && poetry run poe types

# Run everything (strict mode)
poetry run poe all-strict
```

### Step 5: Commit and Push

```bash
# Stage changes
git add dctools/  # Or specific files
git add tests/

# Commit with descriptive message
git commit -m "Add feature: xyz"
# Or for fixes:
git commit -m "Fix: issue with xyz"

# Push to your fork
git push origin feature/my-feature
```

### Step 6: Submit a Pull Request

1. **Go to GitHub** and create a PR from your branch to `main`
2. **Fill in the PR template** with:
   - Clear description of changes
   - Link to related issue(s)
   - Type of change (feature/fix/docs)
   - Testing performed
   - Screenshots (if relevant)

3. **Ensure CI passes** - All tests must pass

4. **Address review comments** - Respond to reviewer feedback

5. **Merge** - Once approved, the PR will be merged

## Code Standards

### Python Versions

- Minimum: Python 3.11
- Maximum: Python < 3.14 (as defined in pyproject.toml)

### Linting

Code is checked with `ruff`:

```bash
# Check linting issues
poetry run poe lint

# Auto-fix issues
poetry run ruff check --fix dctools/
```

### Type Checking

Type hints are required and checked with `mypy`:

```bash
# Check types
poetry run poe types
```

### Testing

We use `pytest` with markers for test organization:

```bash
# Run unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# Run slow tests
pytest -m slow

# Run with coverage
pytest --cov=dctools --cov=dc2
```

#### Test Categories

| Marker | Use Case | Speed | When to run |
|--------|----------|-------|------------|
| `unit` | Isolated unit tests | < 1s | Always (CI) |
| `integration` | Multiple components | 5-30s | Before PR |
| `slow` | Long-running tests | > 30s | Optional, nightly |

### Documentation

- Use **[NumPy docstring style](https://numpydoc.readthedocs.io/)**
- Include type hints in function signatures
- Add examples in docstrings when helpful
- Update documentation files for public API changes

### Imports

Imports are organized and checked:

```bash
# Imports are organized as: stdlib → third-party → local
# This is enforced by ruff
```

Correct import order:
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import xarray as xr

# Local
from dctools.data import CoordinateSystem
```

## Documentation

### Adding Documentation

1. **For new public functions**: Add to API reference (auto-generated from docstrings)
2. **For new modules**: Add page to `docs/source/package_docs/`
3. **For usage patterns**: Add to `docs/source/usage/`
4. **For examples**: Add to Quick Start or specific guide

### Building Documentation Locally

```bash
cd docs
make clean html
open build/html/index.html
```

### Documentation Standards

- Use Markdown (`.md`) files
- Link to other docs using relative paths
- Include code examples for complex features
- Cross-reference related sections

## Commit Guidelines

**Good commit messages**:
- Start with a verb: "Add", "Fix", "Update", "Refactor"
- Be specific: Not "Fixed bug", but "Fix coordinate normalization with lat_c alias"
- Keep under 72 characters for subject line
- Reference issues: "Fix #123" or "Closes #456"

**Example**:
```
Add CoordinateSystem.detect_aliases() method

Automatically detects coordinate aliases (lat_c, nav_lat, etc)
instead of requiring manual specification. Fixes #234.
```

## PR Review Process

We aim to review PRs within 2-3 days. The review will check:

1. **Code quality**: Style, types, tests
2. **Functionality**: Does it solve the problem?
3. **Performance**: Any performance regressions?
4. **Documentation**: Are changes documented?
5. **Backward compatibility**: Any breaking changes?

### Common Feedback

- "Add type hints to this function"
- "This needs a test case"
- "Please update the docstring"
- "Consider moving this to a separate function"
- "This breaks backward compatibility"

## Release Process

The maintainers follow this process for releases:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` (if maintained)
3. Create a git tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. Build and publish to PyPI

## Getting Help

- **Questions about usage**: Open a GitHub Discussion
- **Questions about contributing**: Open an issue with `question` label
- **Need to discuss privately**: Contact maintainers directly

## Community

- **GitHub Issues**: Report bugs and suggest features
- **GitHub Discussions**: Community Q&A
- **PPR Océan & Climat**: Main initiative information

## Acknowledgments

Thank you for contributing! Contributors will be acknowledged:

- In commit history
- In release notes for significant contributions
- In `CONTRIBUTORS.md` (if maintained)

## Resources

- **[Development Setup](docs/source/package_docs/installation.md#method-2-developer-installation)**
- **[Testing Guide](docs/source/package_docs/installation.md#testing)**
- **[Architecture Documentation](docs/source/usage/architecture.md)**
- **[Quick Start Guide](docs/source/usage/quickstart.md)**

---

**Thank you for contributing to dc-tools! 🙏**
