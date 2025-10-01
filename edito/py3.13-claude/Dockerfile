# Image de base fournie par le gestionnaire
FROM inseefrlab/onyxia-jupyter-python:py3.13.7-gpu

# Les variables USERNAME, GROUPNAME, WORKSPACE_DIR sont pré-définies dans l'image de base

# Installation de micromamba
USER root

# Installation des dépendances système nécessaires pour la compilation
RUN rm -f /etc/apt/sources.list.d/ubuntugis-ubuntu-ppa-noble.sources
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libeigen3-dev \
        libblas-dev \
        liblapack-dev \
        cmake \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local bin/micromamba && \
    ln -s /usr/local/bin/micromamba /usr/local/bin/mamba

# Configuration de micromamba pour utiliser un environnement partagé
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_EXE=/usr/local/bin/micromamba

# Création du répertoire /dctools
RUN mkdir -p /dctools 
#&& chown ${USERNAME}:${GROUPNAME} /dctools

# Copie du fichier environment.yml
COPY edito/environment.yml /dctools/environment.yml


# ============
# Installation des dépendances Python dans le système
# ============
# Installation des dépendances micromamba SANS Python
# On utilise -p pour installer directement dans le prefix système
# plutôt que créer un nouvel environnement
# RUN micromamba install -y -p /usr/local -f /dctools/environment.yml && \
#     micromamba clean --all --yes && \
#     rm -rf /tmp/*
# #SHELL ["/bin/bash", "-c"]
# RUN micromamba create -y -n dctools python=3.13.7 && \
#     micromamba install -y -n dctools -f /dctools/environment.yml && \
#     micromamba clean --all --yes && \
#     rm -rf /tmp/*
#     RUN micromamba create -y -n dctools \
#     -c conda-forge \
#     --override-channels \
#     esmpy xesmf \
#     python=3.13.7 \  # Spécifier la version exacte déjà installée
#     && micromamba clean --all --yes


# ============
# Installation des dépendances Python dans dans un env dédié
# ============
# Installation des dépendances micromamba
# On crée l'environnement à partir du fichier environment.yml
# Nettoyage dans la même couche
RUN micromamba create -y -n dctools -f /dctools/environment.yml && \
    micromamba clean --all --yes

# Activation de l'environnement pour les commandes suivantes
SHELL ["/usr/local/bin/micromamba", "run", "-n", "dctools", "/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/dctools/bin:$PATH


# Configuration de Poetry pour ne pas créer de virtualenv
# et installer dans l'environnement système (qui est déjà l'environnement micromamba)
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME=/opt/poetry

# Configuration de l'emplacement des caches de pip and poetry
ENV PIPO_CACHE_DIR=/root/.cache/pip \
    POETRY_CACHE_DIR=/root/.cache/pypoetry

# Installation de poetry (déjà dans environment.yml via micromamba)
# Vérification de la présence de poetry
RUN which poetry || pip install poetry

# Copie des fichiers de configuration Poetry
WORKDIR /dctools
COPY pyproject.toml ./
COPY dctools ./dctools
COPY README.md ./README.md
COPY LICENSE ./LICENSE
#COPY --chown=${USERNAME}:${GROUPNAME} . /app/

# Installation des dépendances avec Poetry
# On utilise --no-root pour ne pas installer le package lui-même à ce stade
# Nettoyage des caches dans la même couche
RUN poetry install --no-root --with custom && \
    poetry cache clear pypi --all -n || true && \
    poetry cache clear _default_cache --all -n || true && \
    pip cache purge || true && \
    rm -rf /root/.cache /home/*/.cache 2>/dev/null || true

# Installation du package 
RUN poetry install --only-root && \
    poetry cache clear pypi --all -n || true && \
    poetry cache clear _default_cache --all -n || true && \
    pip cache purge || true && \
    rm -rf /root/.cache /home/*/.cache 2>/dev/null || true

#Set shell back to sh
#SHELL ["/bin/sh", "-c"]

# Config Jupyter Kernel Jupyter par celui de micromamba
# RUN /opt/python/bin/pip install jupyterlab && \
#     /opt/python/bin/pip install ipykernel && \
#     /opt/python/bin/jupyter kernelspec uninstall -f python3 && \
#     /opt/python/bin/python -m ipykernel install --name=python3 --display-name "Dc-tools" && \
#     rm -rf /root/.cache/pip
# Installation de ipykernel dans l'environnement conda


RUN pip install ipykernel && \
    # Suppression de l'ancien kernel 
    jupyter kernelspec uninstall -f python3 && \
    # Installation du kernel lié à l'environnement conda, dans un emplacement système
    python -m ipykernel install --name=dctools --display-name="Dc-tools" --prefix=/usr/local && \
    # Nettoyage
    micromamba clean --all --yes && \
    rm -rf /root/.cache/pip


# =======================
# FIX : Give user permission on all files in HOME
# =======================
RUN find "${HOME}" -not -user "${USERNAME}" -execdir chown --no-dereference "${USERNAME}:${GROUPNAME}" {} \+


# For security reasons, we do not want to run the container as root
USER ${USERNAME}
WORKDIR ${WORKSPACE_DIR}


# =======================
# Manage Jupyterlab launch
# =======================
# 
# CMD on onyxia-jupyter-python is buggy with WORKSPACE_DIR
#  - ${WORKSPACE_DIR} is not expanded if used directly in CMD
# On Edito datalab Edito, the launch command is overridden anyway in the k8s deployment.yaml file
#     command: ["/bin/sh","-c"]
#     args: ["{{ .Values.init.standardInitPath }} jupyter lab --no-browser --ip '0.0.0.0' --LabApp.token='$(PASSWORD)' --ContentsManager.allow_hidden=True"]
#   "standardInitPath" is defined in values.yaml file : `standardInitPath: /opt/onyxia-init.sh`
# So the command is at last ["/bin/sh","-c", jupyter lab --no-browser --ip '0.0.0.0' --LabApp.token='$(PASSWORD)' --ContentsManager.allow_hidden=True"]
# We use a bash -c command to expand the variable  
# CMD ["jupyter", "lab", "--no-browser", "--ip", "0.0.0.0", "--notebook-dir", "${WORKSPACE_DIR}"]
#CMD ["/bin/sh", "-c", "jupyter lab --no-browser --ip 0.0.0.0 --LabApp.token=password --ContentsManager.allow_hidden=True"]
CMD ["/bin/sh", "-c", "jupyter lab --no-browser --ip 0.0.0.0"]
# --NotebookApp.websocket_ping_interval=90000 --NotebookApp.websocket_ping_timeout=90000

