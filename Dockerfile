# FROM --platform=linux/amd64 anibali/pytorch:1.13.1-cuda11.7-ubuntu22.04
FROM --platform=linux/amd64 cnstark/pytorch:1.13.1-py3.9.16-cuda11.7.1-ubuntu20.04
# FROM --platform=linux/amd64 nvidia/cuda:11.7.1-runtime-ubuntu22.04

# FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive 

USER root
RUN apt-get update && apt-get install -y \
    build-essential gcc git ffmpeg libsm6 libxext6 libgeos-dev openslide-tools


# RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository -y ppa:openslide/openslide && apt-get update && apt-get -y install python3-openslide

COPY resources /opt/app/resources
# RUN apt-get update && apt install -y /opt/app/resources/ASAP-2.2-Ubuntu2204.deb && echo "/opt/ASAP/bin" > /home/user/micromamba/lib/python3.9/site-packages/asap.pth

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN chown -R user:user /opt/app/resources
RUN chown -R user:user /tmp

USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

COPY --chown=user:user config /opt/app/config
COPY --chown=user:user dataset_modules /opt/app/dataset_modules
COPY --chown=user:user extract_feature_utils /opt/app/extract_feature_utils
COPY --chown=user:user inference_utils /opt/app/inference_utils
COPY --chown=user:user models /opt/app/models
COPY --chown=user:user utils /opt/app/utils
COPY --chown=user:user wsi_core /opt/app/wsi_core

RUN pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install -e /opt/app/resources/timm-0.5.4

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
# ENTRYPOINT [ "/bin/bash" ]
