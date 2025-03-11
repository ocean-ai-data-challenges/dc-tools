# Installation

**TODO:** write this section.

## Developer dependencies

First, clone the repository:

```bash
git clone git@github.com:ppr-ocean-ia/dc-tools.git
```

Then, inside the newly-created `dc-tools` directory:

```bash
conda create -n dctools
conda activate dctools
conda install xesmf -c confa-forge
poetry install --with dev
```

The reason for this somewhat hacky installation is [a known bug with `xesmf`](https://github.com/pangeo-data/xESMF/issues/269) which causes import errors.
If you're encountering similar problems consider reinstalling the environment.

## Additional documentation dependencies

If instead you want to modify these documentation pages, follow the steps in the ["Developer dependencies"](#developer-dependencies) section.
Then, modify the last command to use the `--with docs` option:

```bash
poetry install --with docs
```
