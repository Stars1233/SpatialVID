# SpatialVID — Docker guide

This repository includes two Dockerfiles:

- `Dockerfile`: simpler image that installs system `ffmpeg` (apt) and Python requirements. Good for quick CPU-only runs or when the host already has compatible CUDA drivers and you don't need NVENC/libvmaf_cuda.
- `Dockerfile.gpu`: advanced image that compiles `nv-codec-headers`, `libvmaf` (with CUDA), and `FFmpeg` with NVENC/NVDEC and `libvmaf_cuda`. Requires a host with NVIDIA drivers & CUDA and is a large, time-consuming build.

Quick start — default image

```bash
# build
docker build -t spatialvid:latest .

# run (interactive)
docker run --rm -v $(pwd):/workspace -w /workspace -it spatialvid:latest
```

GPU-enabled build (advanced)

Important: Build `Dockerfile.gpu` on a machine that has NVIDIA drivers installed and (preferably) CUDA available. The build stage uses the CUDA toolkit from the base image and compiles native code; building in an environment without NVIDIA drivers may still succeed but the resulting ffmpeg NVENC may not function on the host.

```bash
# build using the GPU Dockerfile
docker build -f Dockerfile.gpu -t spatialvid:gpu:latest .

# run with GPU access (requires NVIDIA Container Toolkit)
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace -it spatialvid:gpu:latest
```

Build-time ARGs available in `Dockerfile.gpu`

- `NV_CODEC_HEADERS_BRANCH` (default `sdk/12.1`) — branch of `nv-codec-headers` to checkout. Match this to your CUDA toolkit version.
- `FFMPEG_TAG` (default `release/6.1`) — the FFmpeg tag to build.
- `CUDA_BASE_IMAGE` — base CUDA image used for compilation; adjust if you must match a different CUDA toolkit.

Notes and troubleshooting

- Building `Dockerfile.gpu` may take 30+ minutes and produce a large intermediate image. If you only need NVENC-enabled `ffmpeg`, consider using a prebuilt ffmpeg binary with NVENC (many community images exist) and copying `ffmpeg` into a smaller runtime image.
- If `meson`/`ninja` or `libvmaf` compilation fails, check that meson installed via pip is available in PATH and Meson version >= 0.56.1.
- If FFmpeg `configure` fails about codec headers or nvenc, adjust `NV_CODEC_HEADERS_BRANCH` to match the CUDA/driver version.
- If pip installation of scoring/annotation requirements fails inside the container, add the missing system dev packages to the Dockerfile (for example `libopenexr-dev` for OpenEXR) and rebuild.

CI option

- If you prefer, I can add a GitHub Actions workflow that builds `Dockerfile.gpu` on a runner and returns logs/artifacts. This avoids needing Docker locally and helps iterate until the build is reproducible.

If you want me to proceed to build in CI or further refine the Dockerfiles (multi-stage caching, prebuilt ffmpeg binary stage, or a smaller helper image that only compiles and publishes ffmpeg/build artifacts), tell me which and我会继续。````markdown
# SpatialVID Docker usage

This file describes how to build and run the project inside a Docker container. Two images are provided:

````markdown
# SpatialVID Docker usage

This file describes how to build and run the project inside a Docker container. Two images are provided:

- `Dockerfile` (easier): installs system ffmpeg from apt and the Python requirements. Good for CPU-only runs or when you have a matching system CUDA and preinstalled drivers.
- `Dockerfile.gpu` (advanced): builds `nv-codec-headers`, `libvmaf` (with CUDA), and compiles FFmpeg with NVENC/NVDEC and `libvmaf_cuda`. This is large and must be built on a host with NVIDIA drivers and CUDA.

Prerequisites

- Docker installed. For GPU support, install NVIDIA Container Toolkit and ensure the host has NVIDIA drivers and CUDA available. See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- Sufficient disk space (tens of GB) and CPU time for compiling FFmpeg and libvmaf.

Build the default (simpler) image:

```bash
docker build -t spatialvid:latest .
```

Run container (interactive):

```bash
docker run --rm -v $(pwd):/workspace -w /workspace -it spatialvid:latest
```

Run with docker-compose:

```bash
docker compose up --build
```

GPU-enabled build (full NVENC + libvmaf_cuda)

1. Build the GPU image (must run on machine with NVIDIA drivers and CUDA):

```bash
docker build -f Dockerfile.gpu -t spatialvid:gpu:latest .
```

Notes:

- Building `Dockerfile.gpu` compiles libvmaf and FFmpeg from source and will take a long time (often >30 minutes depending on CPU cores) and produce a large intermediate image. Use `--cpus` and `--memory` docker build options if you need to control resources.
- If you only need NVENC-enabled ffmpeg, you may prefer to use a prebuilt ffmpeg image with NVENC support and copy `ffmpeg` into a simpler runtime image.
- `Dockerfile.gpu` exposes ARGs to match versions (CUDA, nv-codec-headers branch, FFmpeg tag). Edit ARGs at build time if your host uses different CUDA versions.

Example run with GPU access (NVIDIA Container Toolkit):

```bash
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace -it spatialvid:gpu:latest
```

Troubleshooting & tips

- If `meson` or `ninja` version issues occur during libvmaf build, ensure pip-installed meson is on the PATH. The `Dockerfile.gpu` installs `meson`/`ninja` and sets them up for the build stage.
- If FFmpeg's `configure` complains about codec headers or nvenc, double-check `nv-codec-headers` branch matches CUDA toolkit versions. Use the `NV_CODEC_HEADERS_BRANCH` ARG when building.
- If Python packages in `requirements_scoring.txt` still fail to install inside the container, identify missing system packages and add them to the relevant Dockerfile (for example `libopenexr-dev` for OpenEXR). The best approach is iterative: try the build, read the failure, add system package, rebuild.

If you want, I can:

- Try to build the GPU image in CI (GitHub Actions) and return logs so we can iterate without requiring you to install Docker locally.
- Produce a smaller helper image that only builds FFmpeg/libvmaf and publishes the binaries for reuse in a runtime image.

````
# SpatialVID Docker usage

This file describes how to build and run the project inside a Docker container.

Prerequisites

- Docker installed. For GPU support, install NVIDIA Container Toolkit and use `--gpus` when running.
- Sufficient disk space to install pip packages and optional compiled components (libvmaf/ffmpeg) if you choose to compile them.

Build image (CPU or with system CUDA libs):

```bash
docker build -t spatialvid:latest .
```

Run container (interactive):

```bash
docker run --rm -v $(pwd):/workspace -w /workspace -it spatialvid:latest
```

Run with docker-compose:

```bash
docker compose up --build
```

Notes and advanced topics

- The provided Dockerfile installs `ffmpeg` from apt which typically lacks NVENC/NVDEC and libvmaf_cuda. If you need NVIDIA accelerated ffmpeg with libvmaf and NVENC, follow the instructions in `scoring/motion/INSTALL.md` to build FFmpeg with NVIDIA support, or use a prebuilt image that bundles NVENC-enabled ffmpeg.
- Python packages in `requirements/` contain CUDA-specific torch wheels pinned to a particular CUDA version (cu126). If your system GPU/CUDA driver differs, you may need to adjust the base image or the pinned torch wheels.
- If installation of `requirements_scoring.txt` or `requirements_annotation.txt` fails due to packages that require system libs (e.g., `av`, `libvmaf`), consider building those components in a dedicated build stage or using an image that already provides them.
