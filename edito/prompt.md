

Je déploie dans un environnement k8s/docker, une application qui s'appuie sur des GPU Nvidia
Cette application est gérée en utilisant poetry.

Ce sont les développeurs qui utilisent poetry, et qui donc sont en charge du fichier pyproject.yml.
Je souhaterais donc éviter de le modifier.

L'application utilise également des bibliothèques qui n'existent pas sous forme de paquets pypi.
Et que je n'ai par réussi à compiler.
Pour les installer, j'utilise micromamba. et un fichier environment.yml

Je ne peux pas partir d'image de base NVIDIA.
Je dois partir d'images "clé en main" fournies par le gestionnaire de l'environnement de déploiement, que je peux enrichir : 
- environnement CUDA 12.6 + Jupyter + python 3.13.7
- environnement CUDA 12.6 + Torch + Jupyter + python 3.13.7
Micromamba n'est pas installé par défaut sur ces images.
Mais les requirements de mon application imposent python 3.12, je dois donc gérer 2 version de python.



J'ai fait plusieurs tests, et pour les images que j'ai réussi à builder, j'ai recontré plusieurs problèmes.
- L'image buildée demande un environnement CUDA 12.4, mais le serveur necessite une version 12.6
- L'image buildée fait 30Go, car les paquets sont installé en doublon ente le système, l'environnement micromamba, et l'environnement poetry.
- netcdf4 necessite une bibliothèque mpi, et impose une version de python 3.12 au max
- certain paquets comme pyinterp et xbatcher, necessitent un environnement de compilation plus moderne que celui présent, notament Boost 1.79, blas, ...
- la version maximum du cudatoolkit dans micromamba est 11.8, ce qui est insuffisant
- ...

Pour l'environnrnt de runtime du contenaur, il y a 2 contraintes: 
- il faut qu'il se lance en tant que `USER ${USERNAME}` et pas root
- Les Variable £USERNAME, $GROUPNAME, $WORSPACE_DIR sont pré-définies dans l'image de base
- le systéme de déploiment impose la `CMD`, pour lancer jupyter. A ce jour c'est `CMD ["/bin/sh", "-c", "jupyter lab --no-browser --ip 0.0.0.0 --LabApp.token=password --ContentsManager.allow_hidden=True"]`, mais ca évolue. il n'est donc pas possible de modifier CMD

Il faut également limter la taille de l'image autour de 20go max.
Je voudrais avoir une image fonctionnelle dans un premier temps, puis travailler sur la réduction de sa taille ensuite, par exemplee nutilisant un build multi-stage

Je tourne en rond, et ne trouve pas de bonne solution. peux-tu m'aider ?

Pour information, ci dessous mon fichier environment.yml pour micromamba
```
  - nodefaults
channel_priority: strict
dependencies:
  - python(">=3.11.0,<3.14.0")
  - python=3.12.11
  - esmpy
  - xesmf
  - poetry
```
Et mon fichier pypriject.toml
```
[project]
name = "dctools"
description = "Basic tools common to all data challenges."
readme = "README.md"
authors = [
    {name="Kamel Ait Mohand"},
    {name="Guillermo Cossio"},
    ]
license = "GPL-3-only"
keywords = ["xarray", "sea-surface-height", "ocean", "benchmark"]
dynamic = [
    "version",
    "classifiers",
    ]

requires-python = ">=3.11.0,<3.14.0"

dependencies = [
    "argopy>=1.1.0",
    "cartopy>=0.24.1",
    "copernicusmarine>=2.2.2",
    "dask>=2025.7.0",
    "dask[distributed]>=2025.01.0",
    "ftputil (>=5.1.0,<6.0.0)",
    "geopy (>=2.4.1,<3.0.0)",
    "geopandas (>=1.0.1,<2.0.0)",
    "h5py>=3.13.0",
    "json-handler>=2.0.1",
    "loguru>=0.7.3",
    "memory-profiler>=0.61.0",
    "netcdf4 (>=1.5.8,<=1.6.5)", # (>=1.7.2,<2.0.0)",
    "nbformat>=5.10.4",
    "numpy (>=1.25,<2.0.0)",
    "python-json-logger>=3.3.0",
    "pyarrow",
    "pyinterp>=2025.3.0",
    "rich>=13.9.4",
    "s3fs>=2024.10.0",
    "shapely>=1.8.4",
    "tabulate",
    "torch (>=2.6.0,<3.0.0)",
    "torchvision (>=0.21.0,<0.22.0)",
    "torchgeo>=0.6.2",
    "xarray>=2024",
    "xbatcher>=0.4.0",
    "xskillscore>=0.0.27",
    "zarr (>=2.14.2,<3.0.0)",
    "scipy (>=1.13.1,<2.0.0)",
    "dill (>=0.4.0,<0.5.0)"
]

[project.urls]
Repository = "https://github.com/ppr-ocean-ia/dc-tools/tree/main"

[tool.poetry]
version = "0.0.1"

[tool.poetry.group.dev]
optional = true

[tool.poetry.group.dev.dependencies]
poethepoet = "^0.33.0"
pytest = "^7.3.1"
pytest-cov = "^6.0.0"
ruff = "^0.9.10"
mypy = "*"
wget = "^3.2"
patool = "^3.1.3"

[tool.poetry.group.custom]
optional = false

[tool.poetry.group.custom.dependencies]
xrpatcher = { git = "https://github.com/jejjohnson/xrpatcher.git", branch = "main" }
xesmf = { git = "https://github.com/pangeo-data/xESMF.git", branch = "master" }

[build-system]
build-backend = "poetry.core.masonry.api"
requires = ["poetry-core", "setuptools", "wheel", "cmake"]

[tool.ruff]
line-length = 100
indent-width = 4
exclude = [
    ".eggs",
    ".git",
    ".ipynb_checkpoints",
    ".mypy_cache",
    ".nox",
    ".pyenv",
    ".pytest_cache",
    ".pytype",
    ".ruff_cache",
    ".tox",
    ".venv",
    ".vscode",
    "__pypackages__",
    "_build",
    "buck-out",
    "build",
    "dist",
    "docs",
    "node_modules",
    "site-packages",
    "venv",
]

[tool.ruff.lint]
# E - pycodestyle subsets:
		# E4 - whitespace
		# E7 - multiple-statements
		# E9 - trailing-whitespace
# F - Enable Pyflakes
# B - Enable flake8-bugbear
# W - Enable pycodestyle
# C901 - complex-structure
# D - Enable flake8-docstrings
# E501 - line-too-long
select = ["D", "W", "F", "E", "B"]
ignore = ["D401", "D406", "D407"]

[tool.ruff.lint.pydocstyle]
convention = "numpy"  # Accepts: "google", "numpy", or "pep257".

[tool.mypy]
python_version = "3.13"
disable_error_code = ["import-untyped"]
check_untyped_defs = true
ignore_missing_imports = false
warn_unused_ignores = true
warn_redundant_casts = true
warn_unused_configs = true

[tool.poe.tasks]
types = "mypy --install-types --non-interactive dctools"
lint = "ruff check 'dctools' 'tests'"
test = "pytest --capture=no --cov=dctools --cov-fail-under=80 tests/"
# run all
all = [ {ref="lint"}, {ref="types"}, {ref="test"} ]

```