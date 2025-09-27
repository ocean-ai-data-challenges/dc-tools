ARG BASE_IMAGE=ghcr.io/ppr-ocean-ia/dc-tools:poetry-latest
FROM $BASE_IMAGE

LABEL maintainer="emmanuel.braux@imt-atlantique.fr"
LABEL org.opencontainers.image.source="https://github.com/ppr-ocean-ia/dc-tools"
LABEL org.opencontainers.image.description="DC-TOOLS image for GPU workloads on Edito Datalab"

# python env must be configured for $USER
USER ${USERNAME}

# Install DC-TOOLS dependencies with poetry
ENV POETRY_HTTP_TIMEOUT=300

RUN mkdir -p ${HOME}/dctools
WORKDIR ${HOME}/dctools

COPY pyproject.toml ./
COPY dctools ./dctools
COPY README.md ${HOME}/dctools/README.md
COPY LICENSE ${HOME}/dctools/LICENSE

# Install DC-TOOLS
# dependencies frome mamba and poetry are already installed in the BASE_IMAGE
RUN \
    # Install DC-Tools
    poetry install --without dev  --verbose && \
    # Clean caches
    poetry cache clear pypi --all -n && \
    pip cache purge || true && \
    rm -rf /home/$USER/.cache/pypoetry /home/$USER/.cache/pip

RUN python -c "import dctools; print('Import réussi pendant le build !')" || (echo "Erreur d'import de dctools pendant le build !" && exit 1)

# =======================
# FIX : Give user permission on all files in HOME
# =======================
USER root
RUN find "${HOME}" -not -user "${USERNAME}" -execdir chown --no-dereference "${USERNAME}:${GROUPNAME}" {} \+
USER ${USERNAME}


# For security reasons, we do not want to run the container as root
USER ${USERNAME}
WORKDIR ${WORKSPACE_DIR}