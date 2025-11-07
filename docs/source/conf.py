# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PPR-OC Data Challenges'
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
# Mock oceanbench and xrpatcher to prevent errors in readthedocs
autodoc_mock_imports = [
    "oceanbench",
    "xesmf",
    "xrpatcher",
]

# Stolen from weatherbench2:
# https://stackoverflow.com/a/66295922/809705
autosummary_generate = True

# MyST Options
# https://myst-parser.readthedocs.io/en/latest/configuration.html

myst_heading_anchors = 2
myst_links_external_new_tab = True
myst_enable_extensions = [
    "dollarmath"
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
#html_logo = "_static/Logo_PPR.jpg" # TODO: draw a logo for the DCs