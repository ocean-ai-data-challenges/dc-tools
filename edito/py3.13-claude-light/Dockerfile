# ============================================
#  Multi stage Build :
#    - Stage 1: Builder - Installation and compilation
#    - Stage 2: Runtime - Final minimal image
# ============================================


# ============================================
# Stage 1: Builder - Installation et compilation
# ============================================
ARG BASE_IMAGE=inseefrlab/onyxia-python-minimal:py3.13.7-gpu
FROM ${BASE_IMAGE} as builder

# Environment variables inherited from the base image : 
# USERNAME, GROUPNAME, WORKSPACE_DIR

# Change to root to install system dependencies
USER root

# Clean BASE Image
RUN \
    # Clean workspace
    rm -rf /home/onyxia/work/* && \
    # Remove ubuntugis PPA, which is not available for Ubuntu Noble
    rm -f /etc/apt/sources.list.d/ubuntugis-ubuntu-ppa-noble.sources

# Installation of system dependencies needed for compilation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libeigen3-dev \
        # Package for BLAS library (if auto-detection fails)
        libblas-dev \
        liblapack-dev \
        cmake \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Micromamba installation
ENV MAMBA_BIN_PATH=/usr/local/bin
RUN mkdir -p ${MAMBA_BIN_PATH} &&\
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ${MAMBA_BIN_PATH} --strip-components=1 &&\
    chmod +x ${MAMBA_BIN_PATH}/micromamba && \
    ln -s /usr/local/bin/micromamba /usr/local/bin/mamba
    
# Micromamba configuration to use a shared environment
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_EXE=/usr/local/bin/micromamba

# Creation of /dctools directory
RUN mkdir -p /dctools 
#&& chown ${USERNAME}:${GROUPNAME} /dctools

# ============
# Python dependencies installation in a dedicated environment
# ============
# Micromamba dependencies installation from the environment.yml file
# Jupyterlab and ipykernel are added to be able to use the environment in Jupyterlab
# Cleanup in the same layer
COPY edito/environment.yml /dctools/environment.yml
RUN micromamba create -y -n dctools -f /dctools/environment.yml && \
    # Add Jupyterlab and ipykernel to be able to use the environment in Jupyterlab
    micromamba install -n dctools jupyterlab ipykernel -y && \
    micromamba clean --all --yes

# Environment activation for following commands
SHELL ["/usr/local/bin/micromamba", "run", "-n", "dctools", "/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/dctools/bin:$PATH


# Poetry configuration to not create virtualenv
# and install in the system environment (which is already the micromamba environment)
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME=/opt/poetry

# Configuration of pip and poetry cache location
ENV PIPO_CACHE_DIR=/root/.cache/pip \
    POETRY_CACHE_DIR=/root/.cache/pypoetry

# Poetry installation (already in environment.yml via micromamba)
# Check for poetry presence
RUN which poetry || pip install poetry

# Copy Poetry configuration files
WORKDIR /dctools
COPY pyproject.toml ./
COPY dctools ./dctools
COPY README.md ./README.md
COPY LICENSE ./LICENSE
#COPY --chown=${USERNAME}:${GROUPNAME} . /app/

# Dependencies installation with Poetry
#   - use --no-root to not install the package itself at this stage
#   - use --without dev to not install dev dependencies
#   - use --verbose to have more details during installation
# Use SSH mount to access private repositories if needed (oceanbench)
# Note: --mount=type=ssh requires BuildKit, and `DOCKER_BUILDKIT=1` environment variable set
# Note: SSH keys must be added to the docker build command with `--ssh` option
# Example: `docker build --ssh default -t myimage .`
# Cache cleanup in the same layer
RUN \
    # Manage SSH key for git+ssh
    --mount=type=ssh \
    mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh && \
    # get github.com SSH key to avoid host verification prompt
    ssh-keyscan github.com >> /root/.ssh/known_hosts && \
    # Build perf
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    # Install dependencies without installing the package itself
    poetry install --no-root  --without dev --verbose && \
    # Cleanup
    poetry cache clear pypi --all -n || true && \
    poetry cache clear _default_cache --all -n || true && \
    pip cache purge || true && \
    rm -rf /root/.cache /home/*/.cache 2>/dev/null || true

   
# Package installation as "non-editable"
#  - Separate layer to be able to rebuild only this layer if needed
RUN \
    # Build perf
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/pypoetry \
    # Build the package as a wheel
    poetry build -f wheel && \
    # Install the wheel (ensures a non-editable installation)
    pip install --no-deps --verbose dist/*.whl && \
    # Cleanup
    poetry cache clear pypi --all -n || true && \
    poetry cache clear _default_cache --all -n || true && \
    rm -rf dist/ build/ && \
    pip cache purge || true && \
    rm -rf /root/.cache /home/*/.cache 2>/dev/null || true

# --- 
# Aggressive cleanup to reduce image size
# --- 
# remove Micromamba cache 
RUN micromamba clean --all --yes && \
    rm -rf /opt/conda/pkgs /opt/conda/conda-meta

# Remove Poetry and pip build dependencies which are not needed at runtime
RUN pip uninstall -y poetry pip setuptools wheel && \
    rm -rf /opt/poetry

# DCtools environment cleanup
RUN \
    # Python cache files
    find /opt/conda/envs/dctools -name "*.pyc" -delete && \
    find /opt/conda/envs/dctools -name "*.pyo" -delete && \
    find /opt/conda/envs/dctools -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    # Supprimer les tests
    find /opt/conda/envs/dctools -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/conda/envs/dctools -name "test" -type d -exec rm -rf {} + 2>/dev/null || true && \
    # Cleanup du .pth problématique
    rm -f /opt/conda/envs/dctools/lib/python3.13/site-packages/distutils-precedence.pth && \
    # Supprimer les fichiers de documentation
    rm -rf /opt/conda/envs/dctools/share/doc \
           /opt/conda/envs/dctools/share/man \
           /opt/conda/envs/dctools/share/gtk-doc && \
    # Supprimer les headers et librairies statiques (si pas besoin de compilation)
    rm -rf /opt/conda/envs/dctools/include \
           /opt/conda/envs/dctools/lib/*.a && \
    # Strip les binaires
    find /opt/conda/envs/dctools -name "*.so" -exec strip --strip-unneeded {} \; 2>/dev/null || true && \
    # NE PAS supprimer les .dist-info et .egg-info (nécessaires pour l'introspection des packages)
    # find /opt/conda/envs/dctools -name "*.a" -delete 2>/dev/null || true && \
    # find /opt/conda/envs/dctools -type d -name "*.dist-info" -exec rm -rf {}/RECORD {} + && \
    # find /opt/conda/envs/dctools -type d -name "*.egg-info" -exec rm -rf {} + && \
    rm -rf /tmp/* /var/tmp/*

# =======================
# FIX : Give user permission on all files in HOME
# =======================
RUN find "${HOME}" -not -user "${USERNAME}" -execdir chown --no-dereference "${USERNAME}:${GROUPNAME}" {} \+

# For security reasons, we do not want to run the container as root
USER ${USERNAME}
WORKDIR ${WORKSPACE_DIR}



# ============================================
# Stage 2: Runtime - Image finale minimale
# ============================================
FROM ${BASE_IMAGE} as runtime

# Environment variables inherited from the base image : 
# USERNAME, GROUPNAME, WORKSPACE_DIR

USER root

# Remove ubuntugis PPA, which is not available for Ubuntu Noble
RUN rm -f /etc/apt/sources.list.d/ubuntugis-ubuntu-ppa-noble.sources

# Installation uniquement des dépendances runtime (pas les headers de dev)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libboost-filesystem1.83.0 \
        libboost-system1.83.0 \
        libblas3 \
        liblapack3 \
        ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \
        /tmp/* \
        /var/tmp/* \
        /usr/share/man \
        /usr/share/doc \
        /usr/share/doc-base

# Copy ONLY the installed Python environment (not micromamba, not the source code)
COPY --from=builder /opt/conda/envs/dctools /opt/conda/envs/dctools

# Configuration complète de l'environnement Python
ENV PATH=/opt/conda/envs/dctools/bin:$PATH \
    PYTHONPATH=/opt/conda/envs/dctools/lib/python3.13/site-packages:$PYTHONPATH \
    LD_LIBRARY_PATH=/opt/conda/envs/dctools/lib:$LD_LIBRARY_PATH

# Permissions on HOME files
RUN find "${HOME}" -not -user "${USERNAME}" -execdir chown --no-dereference "${USERNAME}:${GROUPNAME}" {} \+ 2>/dev/null || true

# Aggressive final cleanup
RUN find /opt/conda/envs/dctools -name "*.pyc" -delete && \
    find /opt/conda/envs/dctools -name "*.pyo" -delete && \
    find /opt/conda/envs/dctools -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/conda/envs/dctools -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/conda/envs/dctools -name "*.a" -delete 2>/dev/null || true && \
    rm -rf /tmp/* /var/tmp/*



# For security reasons, we do not want to run the container as root
USER ${USERNAME}
WORKDIR ${WORKSPACE_DIR}

# =======================
# Manage Jupyterlab launch
# =======================
# CMD on onyxia-jupyter-python is buggy with WORKSPACE_DIR
#  - ${WORKSPACE_DIR} is not expanded if used directly in CMD
# On Edito datalab Edito, the launch command is overridden anyway in the k8s deployment.yaml file
#     command: ["/bin/sh","-c"]
#     args: ["{{ .Values.init.standardInitPath }} jupyter lab --no-browser --ip '0.0.0.0' --LabApp.token='$(PASSWORD)' --ContentsManager.allow_hidden=True"]
#   "standardInitPath" is defined in values.yaml file : `standardInitPath: /opt/onyxia-init.sh`
# So the command is at last ["/bin/sh","-c", jupyter lab --no-browser --ip '0.0.0.0' --LabApp.token='$(PASSWORD)' --ContentsManager.allow_hidden=True"]
CMD ["/bin/sh", "-c", "jupyter lab --no-browser --ip 0.0.0.0"]
# --NotebookApp.websocket_ping_interval=90000 --NotebookApp.websocket_ping_timeout=90000

