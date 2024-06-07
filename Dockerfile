# FROM --platform=linux/amd64 pytorch/pytorch
FROM --platform=linux/amd64 pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER root

WORKDIR /opt/app

COPY --chown=root:root requirements.txt /opt/app/

COPY --chown=root:root resources /opt/app/resources

COPY --chown=root:root config /opt/app/config
COPY --chown=root:root dataset_modules /opt/app/dataset_modules
COPY --chown=root:root extract_feature_utils /opt/app/extract_feature_utils
COPY --chown=root:root inference_utils /opt/app/inference_utils
COPY --chown=root:root models /opt/app/models
COPY --chown=root:root utils /opt/app/utils
COPY --chown=root:root wsi_core /opt/app/wsi_core

# You can add any Python dependencies to requirements.txt
RUN apt-get update && apt-get install -y \
    gcc ffmpeg libsm6 libxext6 openslide-tools

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY --chown=root:root inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
# ENTRYPOINT [ "/bin/bash" ]
