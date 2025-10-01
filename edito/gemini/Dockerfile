# --- STAGE 1 : BUILDER MMAMBA/POETRY ---
# Utiliser une image de base avec un environnement de compilation moderne
# et Micromamba pré-installé (ou installer Micromamba ici).
# La base doit être assez complète pour satisfaire les exigences de Boost, BLAS, etc.
# Python 3.13.7 est OK ici car on va installer Python 3.12 après avec mamba.
FROM ghcr.io/ppr-ocean-ia/dc-tools:base-latest AS builder


# S'assurer que les variables d'environnement pour Mamba sont définies
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV PATH="${MAMBA_ROOT_PREFIX}/bin:$PATH"

# 1. Installer Micromamba et créer les répertoires racine (NÉCESSITE root)
# On passe temporairement en root
USER root

# CORRECTION : Supprimer les dépôts problématiques qui causent l'échec de apt-get update
# Supprimer le PPA ubuntugis et potentiellement le PGDG (PostgreSQL) si nécessaire.
# Cela cible TOUS les fichiers .list, y compris le principal /etc/apt/sources.list
# RUN sed -i '/ubuntugis\/ppa/d' /etc/apt/sources.list \
#     && find /etc/apt/sources.list.d/ -type f -name "*.list" -exec sed -i '/ubuntugis\/ppa/d' {} \; \
#     && sed -i '/apt.postgresql.org/d' /etc/apt/sources.list \
#     && find /etc/apt/sources.list.d/ -type f -name "*.list" -exec sed -i '/apt.postgresql.org/d' {} \; \
#     && rm -f /var/lib/apt/lists/*
RUN rm -f /etc/apt/sources.list.d/ubuntugis-ubuntu-ppa-noble.sources

# RUN apt-get update && apt-get install -y curl ca-certificates \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    curl ca-certificates bzip2 \
    # Ajout des dépendances de compilation critiques :
    build-essential \
    cmake \
    git \
    libeigen3-dev \
    # Paquet pour la librairie BLAS (si l'auto-détection échoue)
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Installation de Micromamba
ENV MAMBA_BIN_PATH=/usr/local/bin
RUN mkdir -p ${MAMBA_BIN_PATH} &&\
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ${MAMBA_BIN_PATH} --strip-components=1 &&\
    chmod +x ${MAMBA_BIN_PATH}/micromamba 
    #/usr/local/bin/micromamba shell init -p $MAMBA_ROOT_PREFIX -s bash -y
# curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ${MAMBA_BIN_PATH} --strip-components=1 && \

# Créer la structure minimale nécessaire pour que Micromamba fonctionne
RUN mkdir -p $MAMBA_ROOT_PREFIX \
    && mkdir -p $MAMBA_ROOT_PREFIX/pkgs \
    && mkdir -p $MAMBA_ROOT_PREFIX/envs

# 2. Donner la propriété complète du répertoire racine Mamba à l'utilisateur non-root
RUN chown -R ${UID}:${UID} $MAMBA_ROOT_PREFIX \
    && chmod -R u+rwX,go+rX,go-w $MAMBA_ROOT_PREFIX


# 2. Revenir à l'utilisateur non-root pour les opérations Mamba et Poetry
# L'utilisateur non-root peut maintenant écrire dans /opt/conda
USER ${USERNAME}
WORKDIR /home/${USERNAME}/app

# 3. Création de l'environnement Python 3.12 et le socle Mamba (Python 3.12 et dépendances critiques)
COPY edito/environment.yml /tmp/environment.yml
# # L'environnement sera créé ici: /opt/conda/envs/app_env
RUN micromamba create -p $MAMBA_ROOT_PREFIX/envs/app_env -f /tmp/environment.yml -y \
    && micromamba clean --all --yes

# Définir l'environnement Mamba comme actif
ENV PATH="$MAMBA_ROOT_PREFIX/envs/app_env/bin:$PATH"


# 4. Installer les dépendances Poetry
WORKDIR /app
COPY pyproject.toml ./
COPY dctools ./dctools
COPY README.md ./README.md
COPY LICENSE ./LICENSE

# Error: The current project could not be installed: Readme path `/app/README.md` does not exist.

# Installer les dépendances Poetry dans l'environnement Mamba actif
# --no-root: ne pas installer le paquet local 'dctools' pour l'instant
# --no-dev: ignorer le groupe [dev]
# --no-interaction: pour le build
# Note: Poetry s'installe avec Micromamba, on utilise donc la commande "poetry"
RUN poetry install --no-root --without dev --no-interaction --verbose

# 3. Installer l'application (dctools) après les dépendances
# Ceci permet d'isoler l'installation de l'application elle-même
RUN poetry install --without dev --no-interaction

# # --- STAGE 2 : RUNTIME (Image Minimale pour l'exécution) ---
# # Utiliser l'image runtime officielle pour la meilleure compatibilité CUDA 12.6
# FROM ghcr.io/ppr-ocean-ia/dc-tools:base-latest AS runtime

# # 1. Configurer l'utilisateur (requis par votre système de déploiement)
# # ARG USERNAME=jovyan
# # ARG UID=1000
# #RUN adduser --disabled-password --gecos "" --shell /bin/bash --uid $UID $USERNAME
# USER $USERNAME

# # 2. Copier l'environnement Poetry/Mamba optimisé du builder
# # L'environnement Mamba complet est dans /opt/conda/envs/app_env
# COPY --from=builder $MAMBA_ROOT_PREFIX/envs/app_env $MAMBA_ROOT_PREFIX/envs/app_env

# # Copier le code source
# COPY --from=builder /app /home/$USERNAME/app
# WORKDIR /home/$USERNAME/app

# # 3. Configurer l'environnement d'exécution
# # On s'assure que le PATH pointe vers le Python de Mamba
# ENV MAMBA_ROOT_PREFIX="/opt/conda"
# ENV PATH="$MAMBA_ROOT_PREFIX/envs/app_env/bin:$PATH"

# # La CMD est imposée par votre système de déploiement, donc pas de modification ici.
# # Le système de déploiement lancera:
# # CMD ["/bin/sh", "-c", "jupyter lab --no-browser ..."]
# # Assurez-vous que `jupyter` et ses dépendances sont bien installés par Mamba ou Poetry.