

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
  --build-arg BASE_IMAGE=inseefrlab/onyxia-jupyter-python:py3.13.7-gpu \
  -t ghcr.io/ppr-ocean-ia/dc-tools:base-latest \
  .
```

```
docker push ghcr.io/ppr-ocean-ia/dc-tools:base-latest
```

Base image possibles : 

- inseefrlab/onyxia-jupyter-python:py3.12.9-gpu
- inseefrlab/onyxia-jupyter-python:py3.13.7-gpu
- inseefrlab/onyxia-jupyter-pytorch:py3.12.9-gpu
- inseefrlab/onyxia-jupyter-pytorch:py3.13.7-gpu

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

```
docker push ghcr.io/ppr-ocean-ia/dc-tools:mamba-latest
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

```
docker push ghcr.io/ppr-ocean-ia/dc-tools:poetry-latest
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
## Avec oceanBench

Utilisation d'une clé ssh, transmise en tant que secret
```
DOCKER_BUILDKIT=1 docker buildx build \
  --ssh oceanbench_ssh_key=./secrets/oceanbench-deploy \
  --progress=plain \
  --no-cache \
  -f edito/py3.13-claude/Dockerfile \
  --build-arg BASE_IMAGE=inseefrlab/onyxia-jupyter-python:py3.13.7-gpu \
  -t ghcr.io/ppr-ocean-ia/dc-tools:py3.13-oc \
  .
```

Charger la clé
``` bash
# Vérifier si l'agent tourne
echo $SSH_AUTH_SOCK

# Démarrer l'agent et add key
eval $(ssh-agent); ssh-add ./secrets/oceanbench-deploy

# Vérifier les clés chargées
ssh-add -l
```

lancement du build, utilisant la clé
``` bash
DOCKER_BUILDKIT=1 docker buildx build \
  --ssh default \
  --progress=plain \
  --no-cache \
   -f edito/py3.13-claude/Dockerfile \
  --build-arg BASE_IMAGE=inseefrlab/onyxia-jupyter-python:py3.13.7-gpu  \
  -t ghcr.io/ppr-ocean-ia/dc-tools:py3.13-oc \
  .
```
---
## Publication pour service sur edito

``` bash
docker tag ghcr.io/ppr-ocean-ia/dc-tools:latest ghcr.io/ppr-ocean-ia/dc-tools:edito-gpu-latest

docker tag ghcr.io/ppr-ocean-ia/dc-tools:py3.13-oc ghcr.io/ppr-ocean-ia/dc-tools:edito-gpu-latest
docker push ghcr.io/ppr-ocean-ia/dc-tools:edito-gpu-latest
```

---
## Tester

## La version de torch
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Si on met dan la conf micromanba
```
dependencies:
  #- python(">=3.11.0,<3.14.0")
  - python=3.12.11
  - pytorch>=2.6.0,<3.0.0
```
Alors torch est installé sans support CUDA
``` bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
2.5.1.post108 False None
```  

## fonctionnement de dc-tool
```
python -c "import dctools;"
```

---
##
``` bash
#9 20.89     │  ├─ netcdf4 [1.6.4|1.6.5] would require
#9 20.89     │  │  └─ python [>=3.12,<3.13.0a0 *|>=3.12.0rc3,<3.13.0a0 *], which can be installed;
```


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

``` bash
pyinterp utilise la bibliothèque de templates C++ Eigen pour l'algèbre linéaire, et le processus de compilation via cmake ne trouve pas les fichiers d'en-tête (headers) d'Eigen3 sur votre système de construction (l'image Docker builder).
```

---
## Apt sur les images edito

Les source-list sont fausses ....
``` bash
#5 2.587 Reading package lists...
#5 3.220 W: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
#5 3.220 E: The repository 'https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu noble Release' does not have a Release file.
#5 3.220 W: https://apt.postgresql.org/pub/repos/apt/dists/noble-pgdg/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
#5 ERROR: process "/bin/bash -c apt-get update && apt-get install -y curl ca-certificates     && rm -rf /var/lib/apt/lists/*" did not complete successfully: exit code: 100
------
 > [2/5] RUN apt-get update && apt-get install -y curl ca-certificates     && rm -rf /var/lib/apt/lists/*:
1.863 Get:21 http://archive.ubuntu.com/ubuntu noble-updates/multiverse amd64 Packages [38.9 kB]
1.863 Get:22 http://archive.ubuntu.com/ubuntu noble-updates/main amd64 Packages [1,828 kB]
1.953 Get:23 http://archive.ubuntu.com/ubuntu noble-updates/universe amd64 Packages [1,923 kB]
2.032 Get:24 http://archive.ubuntu.com/ubuntu noble-updates/restricted amd64 Packages [2,483 kB]
2.083 Get:25 http://archive.ubuntu.com/ubuntu noble-backports/main amd64 Packages [49.4 kB]
2.084 Get:26 http://archive.ubuntu.com/ubuntu noble-backports/universe amd64 Packages [33.9 kB]

3.220 W: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
3.220 E: The repository 'https://ppa.launchpadcontent.net/ubuntugis/ppa/ubuntu noble Release' does not have a Release file.
3.220 W: https://apt.postgresql.org/pub/repos/apt/dists/noble-pgdg/InRelease: Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg), see the DEPRECATION section in apt-key(8) for details.
------

```
---
## Oceanbench
```bash
#19 [15/18] RUN poetry install --only-root &&     poetry cache clear pypi --all -n || true &&     poetry cache clear _default_cache --all -n || true &&     pip cache purge || true &&     rm -rf /root/.cache /home/*/.cache 2>/dev/null || true
#19 0.656 Updating dependencies
#19 0.656 Resolving dependencies...
#19 7.090 
#19 7.090 The current project's supported Python range (>=3.11.0,<3.14.0) is not compatible with some of the required packages Python requirement:
#19 7.090   - oceanbench requires Python >=3.12.9, so it will not be installable for Python >=3.11.0,<3.12.9
#19 7.090 
#19 7.090 Because dctools depends on oceanbench (0.0.2) @ git+https://github.com/mercator-ocean/oceanbench.git@main which requires Python >=3.12.9, version solving failed.
#19 7.090 
#19 7.090   * Check your dependencies Python requirement: The Python requirement can be specified via the `python` or `markers` properties
#19 7.090 
#19 7.090     For oceanbench, a possible solution would be to set the `python` property to ">=3.12.9,<3.14.0"
#19 7.090 
#19 7.090     https://python-poetry.org/docs/dependency-specification/#python-restricted-dependencies,
#19 7.090     https://python-poetry.org/docs/dependency-specification/#using-environment-markers
#19 7.090 
#19 7.644 WARNING: No matching packages
#19 7.644 Files removed: 0 (0 bytes)
#19 7.687 warning  libmamba Failed to remove file '/home/onyxia/.cache/mamba/proc/1.json' : Success
#19 DONE 7.7s
```