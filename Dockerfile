# Multi-stage Dockerfile for SpatialVID
# Base image with CUDA runtime matching torch cu126 wheels used in requirements
ARG BASE_IMAGE=nvidia/cuda:12.6.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE} as base

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget git build-essential pkg-config ca-certificates \
    python3 python3-pip python3-venv \
    ffmpeg libsm6 libxext6 libxrender1 libgl1 libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install Python build tools commonly needed (meson/ninja used by libvmaf build steps)
RUN pip3 install --no-cache-dir meson ninja

WORKDIR /workspace

# Copy repository into the image
COPY . /workspace

# Install python requirements - split installs to give clearer build output
RUN pip3 --no-cache-dir install -r requirements/requirements.txt
RUN pip3 --no-cache-dir install -r requirements/requirements_scoring.txt || true
RUN pip3 --no-cache-dir install -r requirements/requirements_annotation.txt || true

# Make entrypoint executable
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENV FFMPEG_PATH=/usr/bin/ffmpeg

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

CMD ["bash"]
