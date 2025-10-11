#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$WORKDIR"

echo "[1/6] Checking docker availability..."
if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found. Please install Docker and (for GPU) NVIDIA Container Toolkit." >&2
  exit 2
fi

# Quick permission check: can we talk to the docker daemon?
if ! docker info >/dev/null 2>&1; then
  echo ""
  echo "ERROR: cannot connect to the Docker daemon. Permission denied or daemon not running."
  echo "Common fixes (pick one):"
  echo "  1) Run the script with sudo (quick, but files may be created as root):"
  echo "       sudo ./scripts/build_gpu_docker.sh"
  echo "  2) Add your user to the 'docker' group and re-login (recommended):"
  echo "       sudo usermod -aG docker \$USER" ; 
  echo "       # then log out and log back in, or run: newgrp docker"
  echo "  3) If Docker is installed as snap, use sudo or follow snap-specific docs."
  echo "  4) Ensure daemon is running: sudo systemctl start docker" 
  echo ""
  exit 3
fi

echo "[2/6] Checking NVIDIA runtime support via a test container (this may fail if no GPU or toolkit)..."
if docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA runtime is available. Will build with GPU support.";
else
  echo "Warning: NVIDIA runtime test failed. Build may still succeed, but runtime NVENC/NNDEC features won't work on this host.";
fi

IMAGE_TAG="spatialvid:gpu-local"

echo "[3/6] Building GPU image with Dockerfile.gpu (this may take a long time)..."
# Pre-pull base images used by Dockerfile.gpu to fail early on network/auth issues
BUILDER_IMAGE="swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04"
RUNTIME_IMAGE="swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04"

echo "Pre-pulling base images to validate network/auth and warm cache:"
retry_pull() {
  local img="$1"
  local i
  for i in 1 2 3; do
    echo "  pull attempt $i for ${img}..."
    if docker pull "${img}"; then
      echo "  pulled ${img}"
      return 0
    fi
    sleep $((i * 2))
  done
  echo "Failed to pull ${img} after retries" >&2
  return 1
}

echo "- builder: ${BUILDER_IMAGE}"
echo "- runtime: ${RUNTIME_IMAGE}"
retry_pull "${BUILDER_IMAGE}" || true
retry_pull "${RUNTIME_IMAGE}" || true

USE_BUILDX_CACHE=${USE_BUILDX_CACHE:-1}
# Prefer buildx/BuildKit if available for better output; otherwise fall back
if docker buildx version >/dev/null 2>&1; then
  echo "docker buildx detected — using BuildKit for build"
  if [ "${USE_BUILDX_CACHE}" -eq 1 ]; then
    echo "Using buildx local cache at ./.buildx-cache"
    # Ensure a builder that supports cache export is available. Some drivers (docker driver) do not support cache export.
    if ! docker buildx inspect spatialvid-builder >/dev/null 2>&1; then
      echo "Creating buildx builder 'spatialvid-builder' (driver=docker-container)..."
      if ! docker buildx create --name spatialvid-builder --driver docker-container --use >/dev/null 2>&1; then
        echo "Warning: creating docker-container builder failed; falling back to cache-less build" >&2
        DOCKER_BUILDKIT=1 docker buildx build --progress=plain -f Dockerfile.gpu -t ${IMAGE_TAG} .
      else
        echo "Builder created and selected: spatialvid-builder"
        DOCKER_BUILDKIT=1 docker buildx build --progress=plain \
          --cache-to type=local,dest=.buildx-cache --cache-from type=local,src=.buildx-cache \
          -f Dockerfile.gpu -t ${IMAGE_TAG} .
      fi
    else
      docker buildx use spatialvid-builder >/dev/null 2>&1 || true
      DOCKER_BUILDKIT=1 docker buildx build --progress=plain \
        --cache-to type=local,dest=.buildx-cache --cache-from type=local,src=.buildx-cache \
        -f Dockerfile.gpu -t ${IMAGE_TAG} .
    fi
  else
    DOCKER_BUILDKIT=1 docker build --progress=plain -f Dockerfile.gpu -t ${IMAGE_TAG} .
  fi
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
