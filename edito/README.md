

Executer les build depuis la racine du projet car des ressources dans les dossiers `.devcontainer` et `dc-tools` sont utilisées

| Image de base Edito | inseefrlab/onyxia-jupyter-python:py3.13.7-gpu | 13.8GB |
| 
| Base for DC-Tools | ghcr.io/ppr-ocean-ia/dc-tools:base-latest | 13.8GB |
| 
| Mamba requirement for DC-Tools | ghcr.io/ppr-ocean-ia/dc-tools:mamba-latest | 15.1GB |
| 
| Poetry  requirement for DC-Tools | ghcr.io/ppr-ocean-ia/dc-tools:poery-latest | 28.6GB |
| 
| DC-Tools | ghcr.io/ppr-ocean-ia/dc-tools:latest | |
---
## Base Image

``` bash
docker build \
  --progress=plain \
  --no-cache \
  -f edito/base/Dockerfile \
  --build-arg BASE_IMAGE=inseefrlab/onyxia-jupyter-python:py3.12.9-gpu \
  -t ghcr.io/ppr-ocean-ia/dc-tools:base-latest \
  .
```

- inseefrlab/onyxia-jupyter-python:py3.12.9-gpu
``` bash
#9 20.89     │  ├─ netcdf4 [1.6.4|1.6.5] would require
#9 20.89     │  │  └─ python [>=3.12,<3.13.0a0 *|>=3.12.0rc3,<3.13.0a0 *], which can be installed;
```
- inseefrlab/onyxia-jupyter-python:py3.13.7-gpu
```
docker push ghcr.io/ppr-ocean-ia/dc-tools:base-latest
```

---
## Mamba dependencies

Rebuilder si le fichier `.devcontainer/environment.yml` est modifié.

``` bash
docker build \
  --progress=plain \
  --no-cache \
  -f edito/mamba/Dockerfile \
  --build-arg BASE_IMAGE=ghcr.io/ppr-ocean-ia/dc-tools:base-latest \
  -t ghcr.io/ppr-ocean-ia/dc-tools:mamba-latest \
  .
```

---
## Poetry dependencies

Rebuilder si le fichier `pyproject.toml` est modifié.

``` bash
docker build \
  --progress=plain \
  --no-cache \
  -f edito/poetry/Dockerfile \
  --build-arg BASE_IMAGE=ghcr.io/ppr-ocean-ia/dc-tools:mamba-latest \
  -t ghcr.io/ppr-ocean-ia/dc-tools:poetry-latest \
  .
```

---
## DC-TOOLS

Rebuilder si des fichiers dans le dossier `dctools` sont modifiés.

``` bash
docker build \
  --progress=plain \
  --no-cache \
  -f edito/dctools/Dockerfile \
  --build-arg BASE_IMAGE=ghcr.io/ppr-ocean-ia/dc-tools:poetry-latest \
  -t ghcr.io/ppr-ocean-ia/dc-tools:latest \
  .
```
``` bash
docker push ghcr.io/ppr-ocean-ia/dc-tools:latest
```
---
## Publication pour service sur edito

``` bash
docker tag ghcr.io/ppr-ocean-ia/dc-tools:latest ghcr.io/ppr-ocean-ia/dc-tools:edito-gpu-latest
docker push ghcr.io/ppr-ocean-ia/dc-tools:edito-gpu-latest
```

---
## Pb install Netcdf4

- Message d'erreur lié à mpi si install ave poetry ou pip
``` bash
#10 11.08 Collecting netcdf4==1.6.5#10 11.15   Downloading netCDF4-1.6.5.tar.gz (764 kB)
#10 11.23      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 765.0/765.0 kB 22.2 MB/s  0:00:00
#10 11.25   Installing build dependencies: started
#10 13.10   Installing build dependencies: finished with status 'done'
#10 13.10   Getting requirements to build wheel: started
#10 13.35   Getting requirements to build wheel: finished with status 'error'
#10 13.35   error: subprocess-exited-with-error
...
#10 13.35       ModuleNotFoundError: No module named 'mpi4py'
```
- Installation avec conda, impose version de python 3.12 au max, et limite les version de netcdf4
``` bash
# The following packages are incompatible
     ├─ netcdf4 >=1.5.8,<=1.6.5 * is installable with the potential options
     │  ├─ netcdf4 [1.5.8|1.6.0|...|1.6.5] would require
     │  │  └─ python >=3.10,<3.11.0a0 *, which can be installed;
     │  ├─ netcdf4 [1.6.0|1.6.1|...|1.6.5] would require
     │  │  └─ python >=3.11,<3.12.0a0 *, which can be installed;
     │  ├─ netcdf4 [1.5.8|1.6.0|...|1.6.5] would require
     │  │  └─ python >=3.8,<3.9.0a0 *, which can be installed;
     │  ├─ netcdf4 [1.5.8|1.6.0|...|1.6.5] would require
     │  │  └─ python >=3.9,<3.10.0a0 *, which can be installed;
     │  ├─ netcdf4 [1.6.4|1.6.5] would require
     │  │  └─ python [>=3.12,<3.13.0a0 *|>=3.12.0rc3,<3.13.0a0 *], which can be installed;
     │  └─ netcdf4 [1.5.8|1.6.0|1.6.1] would require
     │     └─ python >=3.7,<3.8.0a0 *, which can be installed;
```
- `"netcdf4 (>=1.5.8,<=1.6.5)"` prévus, mais `python(">=3.11.0,<3.14.0")` demandé d'autre part. On est donc limité à 
``` bash
     │  ├─ netcdf4 [1.6.0|1.6.1|...|1.6.5] would require
     │  │  └─ python >=3.11,<3.12.0a0 *, which can be installed;
```

Modification de l'environnement mamba pour avoir une version de python 12, et installation de Netcdf4 via mamba.

---
## Environnement de compilation pour les paclage installé via pip/poetry

Au moins 2 package posent problème : pyinterp et xbatcher. pyinterp peut être installa via mamba, mais xbatcher ets installé à partir d'un dépôt git, et n'a pas de version précompilée.

Messages d'erreur recontré :

- `Boost 1.79 est requis, mais seulement Boost 1.74 est installé sur le système.
- librairies eigen, blas introuvables.

D'où l'installation de l'environnement de compilation necessaire avec mamba : paquets `eigen`, `boost-cpp`, `openblas`, `lapack`, `blas`, `cmake`, et `pkg-config`


2 options :

1. installer sur le système
``` bash
sudo apt-get update
sudo apt-get install -y libboost-all-dev cmake build-essential
```
2. utiliser conda
``` bash
micromamba install -c conda-forge boost=1.79
```
