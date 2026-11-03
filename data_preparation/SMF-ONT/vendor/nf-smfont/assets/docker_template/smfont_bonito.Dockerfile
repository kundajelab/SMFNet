FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

LABEL maintainer="Mathias Boulanger <mathias.boulanger@embl.de>"

ARG WF_USER=smfontuser WF_UID=1000 WF_GID=100 ARCH=linux-64 BONITOVER=0.9.1

RUN apt-get update && \
      apt-get install -yq --no-install-recommends ca-certificates sudo python3-dev python3-pip python3-wheel python3-setuptools gcc \
      procps build-essential libhdf5-dev libssl-dev libz-dev wget gzip pigz bc bash && \
      apt-get autoclean && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

ENV CONDA_DIR=/home/smfontuser/conda SHELL=/bin/bash WF_USER=smfontuser WF_UID=1000 WF_GID=100 LANG=C.UTF-8 LC_ALL=C.UTF-8 
ENV HOME=/home/smfontuser PATH=/home/smfontuser/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

#COPY fix-permissions /usr/local/bin/fix-permissions

# RUN chmod a+rx /usr/local/bin/fix-permissions && \
#       echo "auth requisite pam_deny.so" >> /etc/pam.d/su && \
#       sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers && \
#       sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && \
#       useradd -m -s /bin/bash -N -u $WF_UID $WF_USER && \
#       chmod g+w /etc/passwd && fix-permissions $HOME

RUN useradd -m -s /bin/bash -N -u $WF_UID $WF_USER

USER 1000
WORKDIR /home/smfontuser
RUN wget -qO- https://micromamba.snakepit.net/api/micromamba/${ARCH}/latest | tar -xvj bin/micromamba && \
      ./bin/micromamba shell init -s bash -p $CONDA_DIR && \
      . /home/smfontuser/.bashrc && \
      chown $WF_USER:$WF_GID $CONDA_DIR && \
      fix-permissions $CONDA_DIR  && fix-permissions $HOME

ENV MAMBA_EXE=/home/smfontuser/bin/micromamba MAMBA_ROOT_PREFIX=/home/smfontuser/conda

USER 1000
WORKDIR /home/smfontuser

RUN . $CONDA_DIR/etc/profile.d/micromamba.sh && \
      micromamba activate && \
      micromamba install --yes -n base -c bioconda -c conda-forge -c r -c epi2melabs \
      python=3.10.0 \
      htslib>=1.17 \
      samtools>=1.17 \
      tabix>=1.11 \
      pip>=4.65.0 && \
      micromamba clean --all --yes && fix-permissions $CONDA_DIR && fix-permissions $HOME 

ARG MAMBA_DOCKERFILE_ACTIVATE=1

RUN pip install --no-cache --use-pep517 ont-bonito==$BONITOVER && \
      rm -rf $CONDA_DIR/include $CONDA_DIR/lib/python3.*/site-packages/pip && \
      find $CONDA_DIR -name '__pycache__' -type d -exec rm -rf '{}' '+' && \
      bonito download --models && remora model download

USER 1000
WORKDIR /home/smfontuser
ENTRYPOINT []
