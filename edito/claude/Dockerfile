FROM ghcr.io/ppr-ocean-ia/dc-tools:base-latest


# =======================
# Global system configuration
# =======================
USER root

# Install Micromamba on system
ENV MAMBA_BIN_PATH=/usr/local/bin
RUN mkdir -p ${MAMBA_BIN_PATH} && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ${MAMBA_BIN_PATH} --strip-components=1 && \
    chmod +x ${MAMBA_BIN_PATH}/micromamba

# Variables d'environnement pour micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/envs/app/bin:$PATH

# Dépendances système nécessaires pour la compilation
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     cmake \
#     libboost-dev \
#     libblas-dev \
#     liblapack-dev \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# Copier le fichier environment.yml
COPY edito/environment.yml /tmp/environment.yml

# Créer l'environnement avec micromamba
RUN micromamba create -n app -y -f /tmp/environment.yml  && \
    micromamba clean --all --yes


###################################
# POETRY
###################################
# # Activer l'environnement pour les commandes suivantes
# SHELL ["/usr/local/bin/micromamba", "run", "-n", "app", "/bin/bash", "-c"]


# # Configurer poetry pour ne pas créer de virtualenv
# RUN poetry config virtualenvs.create false

# # Copier les fichiers de configuration
# COPY pyproject.toml  .

# # Installer les dépendances avec poetry
# RUN poetry install  --no-root --without dev --no-interaction --verbose

# # Copier le code de l'application
# COPY dctools/ .
# WORKDIR dctools
# # # Installer le package
# # RUN poetry install --without dev --no-interaction --verbose

# # Donner les permissions à l'utilisateur non-root
# #ARG USERNAME=jovyan
# RUN chown -R ${USERNAME}:${GROUPNAME} /opt/conda && \
#     chown -R ${USERNAME}:${GROUPNAME} /home/${USERNAME}

# Passer à l'utilisateur non-root
USER ${USERNAME}
WORKDIR ${WORKSPACE_DIR}

# Si utilisation de bash comme shell dans CMD
# # Configuration du shell pour activer automatiquement l'environnement
# # Ceci garantit que jupyter (lancé par CMD) utilisera l'environnement app
# RUN echo 'eval "$(micromamba shell hook --shell bash)"' >> ~/.bashrc && \
#     echo 'micromamba activate app' >> ~/.bashrc
# Activer l'environnement dans le PATH pour tous les processus
# ENV PATH=/opt/conda/envs/app/bin:$PATH
# ENV CONDA_DEFAULT_ENV=app

# Si utilisation de sh comme shell dans CMD
# CRITIQUE : Configurer les variables d'environnement pour que 
# l'environnement micromamba soit utilisé même avec /bin/sh
ENV PATH=/opt/conda/envs/app/bin:$PATH \
    CONDA_DEFAULT_ENV=app \
    CONDA_PREFIX=/opt/conda/envs/app \
    PYTHONPATH=/opt/conda/envs/app/lib/python3.12/site-packages:$PYTHONPATH


# Ne pas toucher à CMD - elle sera fournie par le système de déploiement
# CMD sera ["/bin/sh", "-c", "jupyter lab --no-browser --ip 0.0.0.0 --LabApp.token=password --ContentsManager.allow_hidden=True"]
