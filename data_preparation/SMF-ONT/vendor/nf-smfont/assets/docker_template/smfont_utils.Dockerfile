FROM mambaorg/micromamba:latest

LABEL maintainer="Mathias Boulanger"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER root
RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
        procps git gcc g++ make \
        zlib1g-dev libbz2-dev liblzma-dev \
        libhts3 libhts-dev \
        libtbb12 libtbb-dev \
        libopenblas-dev liblapack-dev \
        libcurl4-gnutls-dev libssl-dev \
        libxml2-dev libicu-dev \
        libpng-dev libjpeg-dev libtiff-dev \
        libcairo2-dev \
        libfreetype6-dev libfontconfig1-dev \
        libxt-dev libx11-dev \
        default-jre-headless && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER mambauser

ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Copy conda environment files
COPY *.yml /tmp/

# Install all conda environments into base
RUN micromamba install --yes --name base --file /tmp/methylTools_conda.yml && \
    micromamba install --yes --name base --file /tmp/BAMutils_conda.yml && \
    micromamba install --yes --name base --file /tmp/SMF_conda.yml && \
    micromamba install --yes --name base --file /tmp/fast5converter_conda.yml && \
    micromamba clean --all --yes && \
    rm -rf $MAMBA_ROOT_PREFIX/include \
           $MAMBA_ROOT_PREFIX/pkgs \
           $MAMBA_ROOT_PREFIX/lib/python3.*/site-packages/pip && \
    find $MAMBA_ROOT_PREFIX -name '__pycache__' -type d -exec rm -rf '{}' '+' 2>/dev/null || true

ARG MAMBA_DOCKERFILE_ACTIVATE=1

USER root
RUN rm -rf /tmp/*.yml

# Apptainer-specific: Create environment file to ensure PATH is set
RUN mkdir -p /.singularity.d/env && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /.singularity.d/env/91-conda-path.sh && \
    chmod 755 /.singularity.d/env/91-conda-path.sh

USER mambauser

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

CMD ["/bin/bash"]

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
