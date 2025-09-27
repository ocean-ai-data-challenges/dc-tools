ARG BASE_IMAGE=ghcr.io/ppr-ocean-ia/dc-tools:base-latest
FROM $BASE_IMAGE

LABEL maintainer="emmanuel.braux@imt-atlantique.fr"
LABEL org.opencontainers.image.source="https://github.com/ppr-ocean-ia/dc-tools"
LABEL org.opencontainers.image.description="Image with mamba dependencies for GPU workloads on Edito Datalab"


# =======================
# Global system configuration
# =======================
USER root

# Install Micromamba on system
ENV MAMBA_BIN_PATH=/usr/local/bin
RUN mkdir -p ${MAMBA_BIN_PATH} && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ${MAMBA_BIN_PATH} --strip-components=1 && \
    chmod +x ${MAMBA_BIN_PATH}/micromamba


# =======================
# DC-TOOLS configuration
# =======================
# DC-TOOLS Installations are done as user to avoid permission issues
USER ${USERNAME}


# Variables pour micromamba
#  - Chemin utilisé par micromamba pour
#    stocker ses environnements et métadonnées
ENV MAMBA_ROOT_PREFIX=${HOME}/micromamba
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
# - Gestion des timeout
ENV MAMBA_DOWNLOAD_TIMEOUT=600
ENV MAMBA_RETRY_COUNT=5
#ENV CONDA_CHANNEL_ALIAS=https://conda.anaconda.org/

# Install dependencies with micromamba (dans base)
# Get environment file
COPY .devcontainer/environment.yml ${HOME}/environment.yml
# Create base environment using python 3.12 to avoid netcdf4 issues
RUN /usr/local/bin/micromamba install -n base python=3.12 -y && \
    /usr/local/bin/micromamba env update -n base -f ${HOME}/environment.yml &&\
    /usr/local/bin/micromamba clean --all --yes

# Installer les dépendances pour le build des paquet pip via micromamba
RUN micromamba install -c conda-forge -y \
    eigen>=3.3.1 \
    boost-cpp>=1.79 \
    openblas \
    lapack \
    blas \
    cmake \
    pkg-config &&\
    /usr/local/bin/micromamba clean --all --yes

# Configurer les variables d'environnement pour CMake
ENV CMAKE_PREFIX_PATH="$MAMBA_ROOT_PREFIX:$CMAKE_PREFIX_PATH"
ENV Eigen3_DIR="$MAMBA_ROOT_PREFIX/share/eigen3/cmake"
ENV EIGEN3_INCLUDE_DIR="$MAMBA_ROOT_PREFIX/include/eigen3"
ENV PKG_CONFIG_PATH="$MAMBA_ROOT_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

# Install netcdf4 separately to avoid build issues with poetry
RUN /usr/local/bin/micromamba install -c conda-forge "netcdf4>=1.6.4,<=1.6.5" && \
    /usr/local/bin/micromamba clean --all --yes


# use lock file if available
#COPY .devcontainer/mamba-environment.lock.json ${HOME}/mamba-environment.lock.json 
# micromamba env create -n mon_env -f environment.yml --dry-run --json > environment.lock.json
# micromamba env export -n mon_env --no-builds > environment.lock.yml

# # Supprime l'environnement existant (si nécessaire) et le recréée
# RUN /usr/local/bin/micromamba env remove -n base && \
#     /usr/local/bin/micromamba env create -n base -f ${HOME}/mamba-environment.lock.json &&\
#     /usr/local/bin/micromamba clean --all --yes

# For security reasons, we do not want to run the container as root
USER ${USERNAME}
WORKDIR ${WORKSPACE_DIR}
