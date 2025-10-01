# Argument pour l'environnement de base, à définir lors du build
#ARG BASE_IMAGE=inseefrlab/onyxia-jupyter-pytorch:py3.13.7-gpu
ARG BASE_IMAGE=inseefrlab/onyxia-jupyter-python:py3.13.7-gpu
FROM ${BASE_IMAGE}

# Définition des variables d'environnement pour Mamba
ENV MAMBA_USER=${USERNAME}
ENV MAMBA_ROOT_PREFIX="${WORKSPACE_DIR}/micromamba"
ENV CONDA_PKGS_DIRS="${MAMBA_ROOT_PREFIX}/pkgs"

# 1. Installation de Micromamba
# Utilisez Mambaforge ou l'installation binaire de micromamba
# Ceci est une hypothèse basée sur un binaire de micromamba
USER root
RUN rm -f /etc/apt/sources.list.d/ubuntugis-ubuntu-ppa-noble.sources
RUN apt-get update && apt-get install -y \
    curl ca-certificates bzip2 \
    # Ajout des dépendances de compilation critiques :
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    # Paquet pour la librairie BLAS (si l'auto-détection échoue)
    libblas-dev \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

ENV MAMBA_BIN_PATH=/usr/local/bin
RUN mkdir -p ${MAMBA_BIN_PATH} &&\
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ${MAMBA_BIN_PATH} --strip-components=1 &&\
    chmod +x ${MAMBA_BIN_PATH}/micromamba 

ENV PATH="/opt/micromamba/bin:${PATH}"

# 2. Configuration de l'environnement utilisateur pour Micromamba
# Crée le répertoire racine pour micromamba sous l'utilisateur (contrainte USERNAME)
USER ${USERNAME}
#RUN micromamba shell init -p ${MAMBA_ROOT_PREFIX} -s bash

# S'assurer que les fichiers de configuration de l'application sont dans l'image
WORKDIR /app
COPY edito/environment.yml ./
COPY pyproject.toml ./
COPY dctools ./dctools
COPY README.md ./README.md
COPY LICENSE ./LICENSE

# Créer un environnement micromamba basé sur environment.yml
# Note: Poetry est inclus ici pour être disponible dans l'environnement
# IMPORTANT: Créer les répertoires de travail de micromamba sous l'utilisateur ${USERNAME}
# Cela garantit que ${USERNAME} a les permissions d'écriture nécessaires.
RUN echo ${MAMBA_ROOT_PREFIX} && \
    mkdir -p ${MAMBA_ROOT_PREFIX} && \
    echo ${CONDA_PKGS_DIRS} && \
    mkdir -p ${CONDA_PKGS_DIRS}

RUN micromamba create -f /app/environment.yml -n base_env && \
    micromamba install -n base_env --yes poetry && \
    micromamba clean -a

# 3. Activation de l'environnement Conda et installation des dépendances Poetry
# Nous activons l'environnement 'base_env' pour que poetry s'installe dedans
# et que les binaires comme `esmpy` soient disponibles.

# Définition du chemin vers le binaire Python de l'environnement Mamba
ENV CONDA_ENV_PATH="${MAMBA_ROOT_PREFIX}/envs/base_env"
ENV PATH="${CONDA_ENV_PATH}/bin:${PATH}"

# Utilisation de l'environnement Python existant (celui de micromamba)
# Le --no-root force poetry à s'installer dans l'environnement et non à l'isoler
# Le --no-interaction est pour le non-téléchargement des dépendances déjà installées par micromamba
RUN poetry install --no-interaction  --no-root --only main

# Exécuter l'application en tant qu'utilisateur non-root
USER ${USERNAME}

# La CMD est imposée, donc pas de modification ici.
# S'assurer que l'activation Conda est gérée (c'est fait via le PATH ci-dessus)
# La variable PATH est modifiée pour pointer vers l'environnement Conda,
# ce qui garantit que Jupyter et Python proviennent de cet environnement.
# La commande CMD est exécutée.