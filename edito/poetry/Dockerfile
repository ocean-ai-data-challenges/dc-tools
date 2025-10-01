ARG BASE_IMAGE=ghcr.io/ppr-ocean-ia/dc-tools:mamba-latest
FROM $BASE_IMAGE

LABEL maintainer="emmanuel.braux@imt-atlantique.fr"
LABEL org.opencontainers.image.source="https://github.com/ppr-ocean-ia/dc-tools"
LABEL org.opencontainers.image.description="Image with mamba and poetry dependencies for GPU workloads on Edito Datalab"

# python env must be configured for $USER
USER ${USERNAME}

# Install DC-TOOLS dependencies with poetry
ENV POETRY_HTTP_TIMEOUT=300

RUN mkdir -p ${HOME}/dctools
WORKDIR ${HOME}/dctools

COPY pyproject.toml ./

# disable MPI support for netCDF4
# "ModuleNotFoundError: No module named 'mpi4py'"
#ENV HDF5_MPI="OFF"

# # dépendances de pour package build
# USER root
# RUN apt update && \
#     apt-get install -y libboost-all-dev cmake build-essential && \
#     apt-get clean && rm -rf /var/lib/apt/lists/*
# USER ${USERNAME}

RUN  \
    # Configure poetry
    poetry config virtualenvs.create false && \
    poetry config virtualenvs.in-project false && \
    poetry self update --preview && \
    # # Install mpi4py separately to avoid build issues
    # poetry run pip install mpi4py && \
    # # 
    # poetry run pip wheel --no-cache-dir --use-pep517 netcdf4==1.6.5 && \
    # Install DC-TOOLS dependencies --skip netcdf4
    poetry install --no-root  --without dev --verbose && \
    # Clean caches
    poetry cache clear pypi --all -n && \
    pip cache purge || true && \
    rm -rf /home/$USERNAME/.cache/pypoetry /home/$USERNAME/.cache/pip


# =======================
# FIX : Give user permission on all files in HOME
# =======================
USER root
RUN find "${HOME}" -not -user "${USERNAME}" -execdir chown --no-dereference "${USERNAME}:${GROUPNAME}" {} \+
USER ${USERNAME}


# For security reasons, we do not want to run the container as root
USER ${USERNAME}
WORKDIR ${WORKSPACE_DIR}