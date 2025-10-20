# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dc-tools'
copyright = '2025, Kamel Ait Mohand, Guillermo Cossio'
author = 'Kamel Ait Mohand, Guillermo Cossio'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = []

# Autodocs/Autosummary config
autodoc_typehints = 'description'
# Mock imports of pyinterp and numpy to avoid bugs on RtD.
# Issues with C libraries like libeigen3 and libboost prevent compiling the docs
autodoc_mock_imports = [
    "oceanbench",
    "netcdf4",
    "numpy",
    "torch",
    "fsspec",
    "loguru",
    "dask",
    "memory_profiler",
    "xarray",
    "psutil",
    "cftime",
    # "argo_data",
    # "cmems_data",
    # "interpolation",
    # "nadir_data",
    # "args_config",
    # "file_utils",
    # "init_dask",
    # "misc_utils",
    # "xarray_utils"
]

# Stolen from weatherbench2:
# https://stackoverflow.com/a/66295922/809705
autosummary_generate = True

# MyST Options
# https://myst-parser.readthedocs.io/en/latest/configuration.html

myst_heading_anchors = 2
myst_links_external_new_tab = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
#html_logo = "_static/Logo_PPR.jpg" # TODO: draw a logo for the DCs