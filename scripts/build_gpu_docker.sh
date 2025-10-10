#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$WORKDIR"

echo "[1/6] Checking docker availability..."
if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Please install Docker and (for GPU) NVIDIA Container Toolkit." >&2
  exit 2
fi

echo "[2/6] Checking NVIDIA runtime support via a test container (this may fail if no GPU or toolkit)..."
if docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA runtime is available. Will build with GPU support.";
else
  echo "Warning: NVIDIA runtime test failed. Build may still succeed, but runtime NVENC/NNDEC features won't work on this host.";
fi

IMAGE_TAG="spatialvid:gpu:local"

echo "[3/6] Building GPU image with Dockerfile.gpu (this may take a long time)..."
# Prefer buildx/BuildKit if available for better output; otherwise fall back
if docker buildx version >/dev/null 2>&1; then
  echo "docker buildx detected — using BuildKit for build"
  DOCKER_BUILDKIT=1 docker build --progress=plain -f Dockerfile.gpu -t ${IMAGE_TAG} .
else
  echo "docker buildx not detected — falling back to classic docker build"
  docker build -f Dockerfile.gpu -t ${IMAGE_TAG} .
fi

echo "[4/6] Checking ffmpeg version inside built image"
docker run --rm --gpus all ${IMAGE_TAG} ffmpeg -version | sed -n '1,12p' || true

echo "[5/6] Checking for vmaf filter and nvenc encoders"
docker run --rm --gpus all ${IMAGE_TAG} ffmpeg -filters | grep vmaf || echo "vmaf filter not present"
docker run --rm --gpus all ${IMAGE_TAG} ffmpeg -encoders | grep nvenc || echo "nvenc encoders not present"

echo "[6/6] Python smoke test: check python and torch (if installed)"
docker run --rm --gpus all ${IMAGE_TAG} python -c "import sys; print('python', sys.version)" || true
docker run --rm --gpus all ${IMAGE_TAG} python -c "import importlib, sys
try:
    t=importlib.import_module('torch')
    print('torch', t.__version__, 'cuda', getattr(t, 'version', None) and getattr(t.version, 'cuda', lambda: 'unknown')())
except Exception as e:
    print('torch not importable:', e)
" || true

echo "Done. If any step printed warnings or errors, inspect build output above and adjust Dockerfile.gpu (e.g., NV_CODEC_HEADERS_BRANCH, missing -dev packages)."

echo "Helpful next commands:"
echo "  docker images | grep spatialvid"
echo "  docker run --gpus all -it --rm -v \\$(pwd):/workspace -w /workspace ${IMAGE_TAG} bash"
