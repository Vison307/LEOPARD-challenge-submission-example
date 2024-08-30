# FROM --platform=linux/amd64 pytorch/pytorch
FROM --platform=linux/amd64 cnstark/pytorch:1.13.1-py3.9.16-cuda11.7.1-ubuntu20.04
# FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y \
    build-essential gcc ffmpeg libsm6 libxext6 git
RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:openslide/openslide && apt-get update && apt-get -y install openslide-tools


RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

COPY --chown=user:user resources /opt/app/resources

COPY --chown=user:user config /opt/app/config
COPY --chown=user:user dataset_modules /opt/app/dataset_modules
COPY --chown=user:user extract_feature_utils /opt/app/extract_feature_utils
COPY --chown=user:user inference_utils /opt/app/inference_utils
COPY --chown=user:user models /opt/app/models
COPY --chown=user:user utils /opt/app/utils
COPY --chown=user:user wsi_core /opt/app/wsi_core

# You can add any Python dependencies to requirements.txt

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN python -m pip install -e /opt/app/resources/timm-0.5.4
# RUN python -m pip install git+https://gitclone.com/github.com/Mahmoodlab/CONCH.git

COPY --chown=user:user inference.py /opt/app/


ENTRYPOINT ["python", "inference.py"]
# ENTRYPOINT [ "/bin/bash" ]