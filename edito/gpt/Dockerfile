# ---------- Stage 1 : builder ----------
FROM nvidia/cuda:12.6.2-devel-ubuntu22.04 AS builder

# Paquets de base pour compiler
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake git curl wget \
    libhdf5-dev libnetcdf-dev libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Micromamba on system
ENV MAMBA_BIN_PATH=/usr/local/bin
RUN mkdir -p ${MAMBA_BIN_PATH} && \
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C ${MAMBA_BIN_PATH} --strip-components=1 && \
    chmod +x ${MAMBA_BIN_PATH}/micromamba

# créer env de build via environment.yml
WORKDIR /src
COPY edito/environment.yml /src/

RUN micromamba create -y -n runtime -f environment.yml -c conda-forge -c nvidia \
 && micromamba clean --all --yes
 
RUN micromamba create -y -n build -f environment.yml -c conda-forge -c nvidia && \
    micromamba clean --all --yes

# RUN micromamba create -y -n build -c conda-forge \
#     python=3.12 \
#     boost-cpp eigen pybind11 \
#     pip setuptools wheel \
#     && micromamba clean --all --yes


# RUN micromamba install -y -n build -f /src/environment.yml && \
#     micromamba clean --all --yes


SHELL ["micromamba", "run", "-n", "build", "/bin/bash", "-c"]

# Copier fichiers de projet
COPY pyproject.toml  /src/

# Installer poetry dans le builder seulement
#RUN pip install poetry==1.7.1

# Exporter les dépendances résolues
RUN poetry export -f requirements.txt --without-hashes --verbose -o /tmp/requirements.txt

# Construire les wheels (y compris netcdf4<=1.6.5, pyinterp, etc.)
RUN mkdir /wheels && \
    pip wheel -r /tmp/requirements.txt -w /wheels

# ---------- Stage 2 : runtime ----------
#FROM nvidia/cuda:12.6.2-runtime-ubuntu22.04 AS runtime
FROM ghcr.io/ppr-ocean-ia/dc-tools:base-latest AS runtime

# # Installer Python 3.12 et pip
# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
#     python3.12 python3.12-venv python3-pip \
#     && rm -rf /var/lib/apt/lists/*

# Assurer que python pointe vers 3.12
#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

WORKDIR /app

# Copier les wheels compilées
COPY --from=builder /wheels /wheels

# Installer dans l’environnement système (pas de doublon poetry/micromamba)
RUN pip install --no-cache-dir /wheels/*

# Copier ton code
COPY dctools/ /app

ENV PYTHONPATH=/app

# CMD ["python", "-m", "dctools"]
