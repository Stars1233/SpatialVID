# Compiling FFmpeg with NVIDIA GPU Acceleration and VMAF on Ubuntu

This guide provides a comprehensive walkthrough for compiling FFmpeg from source on an Ubuntu system equipped with an NVIDIA GPU. The resulting build will support NVIDIA's hardware encoding/decoding (NVENC/DEC), NPP filters (NVIDIA Performance Primitives), and CUDA-based VMAF (Video Multi-Method Assessment Fusion) for video quality assessment.


## Environment and Versions

Before you begin, ensure your system environment is similar to the configuration below. Version matching is crucial for a successful compilation.
The GPU needs to support HEVC; refer to the [NVIDIA NVDEC Support Matrix](https://en.wikipedia.org/wiki/NVIDIA_Video_Coding_Engine#NVDEC).

-   **GPU**: NVIDIA GeForce RTX 4090 or other compatible models
-   **OS**: Ubuntu 22.04 
-   **NVIDIA Driver Version**: A version compatible with CUDA 12.6
-   **CUDA Version (from `nvidia-smi`)**: `12.x`
-   **CUDA Toolkit Version**: `12.6` (This is the version used for compilation)
-   **Target FFmpeg Version**: `6.1`

**Key Tip**: The version of the `NVIDIA Codec Headers` (`ffnvcodec`) must be compatible with the `CUDA Toolkit` version installed on your system and the version of `FFmpeg` you intend to compile.

## Compilation Steps

Please follow these steps in order.

### Step 1: Install System Dependencies


```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential yasm nasm cmake libtool libc6-dev unzip wget git pkg-config \
    python3 python3-pip ninja-build ca-certificates libnuma-dev \
    libx264-dev libx265-dev libvpx-dev libfdk-aac-dev libmp3lame-dev libopus-dev \
    libass-dev libssl-dev
```
**Note**: The command above installs the development packages for common audio/video codecs like `libx264`, `libx265`, `libvpx` (VP8/VP9), and `libfdk-aac` (AAC), which allows them to be enabled in FFmpeg.

### Step 2: Install Meson

`libvmaf` uses `Meson` and `Ninja` for its build process. We already installed `ninja-build` via `apt` in the previous step. Now, we'll use `pip` to install a newer version of `Meson`.

1.  **Install Meson using pip**

    ```bash
    sudo python3 -m pip install --upgrade pip setuptools wheel
    sudo python3 -m pip install --upgrade 'meson>=0.56.1'
    ```

2.  **Verify the version and handle path issues**
    `libvmaf` requires Meson version ≥ 0.56.1.

    ```bash
    meson --version
    ```

    If the version is too low or the command is not found, it's likely because the newly installed executable is not in your system's `PATH`.

    ```bash
    # Find where meson was installed
    which meson
    # The output is usually /usr/local/bin/meson

    # Add its path to your environment (for the current session)
    export PATH="/usr/local/bin:$PATH"

    # Add it permanently to your ~/.bashrc
    echo 'export PATH="/usr/local/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    ```

### Step 3: Install NVIDIA Codec Headers

These are the header files FFmpeg needs to interface with the NVIDIA driver for hardware-accelerated video processing.

1.  **Clone the repository**

    ```bash
    git clone https://github.com/FFmpeg/nv-codec-headers.git
    cd nv-codec-headers
    ```

2.  **Check out the branch that matches your CUDA Toolkit**
    Since the highest available version of this library is currently 12.2, we will use the `sdk/12.1` branch here.

    ```bash
    git checkout sdk/12.1
    ```

3.  **Install the headers**

    ```bash
    sudo make install
    cd ..
    ```

### Step 4: Compile and Install VMAF (libvmaf)

To use the `libvmaf_cuda` filter in FFmpeg, we must first compile `libvmaf` from source with CUDA support enabled. This will use the latest version from the main branch.

1.  **Clone the VMAF repository**

    ```bash
    git clone https://github.com/Netflix/vmaf.git
    cd vmaf
    ```

2.  **Configure the project with Meson**
    Use the `meson setup` command to configure the build options, making sure to enable CUDA.

    **IMPORTANT**: If you have had a failed compilation attempt before, make sure to clean up the old build directory (`rm -rf libvmaf/build`). Ensure that the CUDA paths below point correctly to your CUDA 12.6 installation (the default `/usr/local/cuda` is usually a symlink to the correct version).

    Using the following command, which explicitly specifies CUDA paths, is highly recommended:

    ```bash
    meson setup libvmaf/build libvmaf \
        --buildtype=release \
        -Denable_cuda=true \
        -Dcuda_include_dir=/usr/local/cuda/include \
        -Dcuda_lib_dir=/usr/local/cuda/lib64 \
        -Dcuda_compiler=/usr/local/cuda/bin/nvcc
    ```

3.  **Compile and install**

    ```bash
    # Compile
    ninja -C libvmaf/build
    # Install
    sudo ninja -C libvmaf/build install
    ```

4.  **Refresh the dynamic library cache**
    This crucial step ensures the system can find the newly installed `libvmaf.so`.

    ```bash
    sudo ldconfig
    cd ..
    ```

### Step 5: Compile and Install FFmpeg

With all dependencies in place, we can now compile FFmpeg itself.

1.  **Clone the FFmpeg repository**

    ```bash
    git clone https://github.com/FFmpeg/FFmpeg.git
    cd FFmpeg
    ```

2.  **Check out the target version**

    ```bash
    git checkout release/6.1
    ```

3.  **Configure the build**
    This command configures FFmpeg to enable NVIDIA features, `libvmaf`, and the other third-party libraries we installed in Step 1. Ensure that the CUDA paths below point correctly to your CUDA 12.6 installation.

    ```bash
    ./configure \
        --enable-nonfree \
        --enable-gpl \
        --enable-cuda-nvcc \
        --enable-libnpp \
        --extra-cflags="-I/usr/local/cuda/include -I/usr/local/cuda/include -I/usr/local/include" \
        --extra-ldflags="-L/usr/local/cuda/lib64 -L/usr/local/cuda/compat" \
        --disable-static \
        --enable-shared \
        --enable-libvmaf \
        --enable-libx265 \
    ```

4.  **Compile and install**
    Use the `j` flag to parallelize the compilation and speed it up (e.g., using `8` cores).

    ```bash
    make -j8
    sudo make install
    ```

5.  **Refresh the dynamic library cache again**
    This allows the system to recognize the newly installed FFmpeg libraries.

    ```bash
    sudo ldconfig
    ```

## Verify the Installation

After installation, you can run these commands to confirm that FFmpeg was built correctly.

1.  **Check version and configuration flags**

    ```bash
    ffmpeg -version
    ```

    The output should include the flags you enabled, such as `--enable-cuda-nvcc`, `--enable-libnpp`, `--enable-libvmaf`, and `--enable-libx264`.

2.  **Check for available hardware accelerators**

    ```bash
    ffmpeg -hwaccels
    ```

    The list should include `cuda`.

3.  **Check for NVIDIA and other encoders**

    ```bash
    # Check for NVIDIA encoders
    ffmpeg -encoders | grep nvenc
    # You should see encoders like h264_nvenc, hevc_nvenc, etc.

    # Check for the libx264 encoder
    ffmpeg -encoders | grep libx264
    # You should see libx264

    # Check for the libfdk_aac encoder
    ffmpeg -encoders | grep libfdk_aac
    # You should see libfdk_aac
    ```

## Usage Example

This command demonstrates how to use hardware acceleration (`scale_npp`) and the CUDA-based VMAF filter to assess video quality.

```bash
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i distorted_video.mp4 -i reference_video.mp4 \
-filter_complex \
"[0:v]scale_npp=format=yuv420p[dis];[1:v]scale_npp=format=yuv420p[ref];[dis][ref]libvmaf_cuda" \
-f null -
```

-   `-hwaccel cuda`: Enables CUDA hardware decoding.
-   `scale_npp`: Uses the NVIDIA NPP library for GPU-accelerated video scaling.
-   `libvmaf_cuda`: Uses the CUDA-based VMAF filter for calculations.

## Troubleshooting

### Issue 1: VMAF compilation fails with `vcs_version.h: No such file or directory`

-   **Cause**: This error typically occurs if you downloaded the VMAF source code as a ZIP archive instead of using `git clone`. The build script relies on the `.git` directory to generate version header files.
-   **Solution**: Always use `git clone` to get the source code.
    ```bash
    git clone https://github.com/Netflix/vmaf.git
    ```

### Issue 2: FFmpeg `configure` fails with error about Video Codec SDK version being too low

-   **Error Message**: Something like `ERROR: nvenc requested, but NVIDIA Video Codec SDK 12.1 or later is required.` (The version number may vary).
-   **Cause**: This means the version of `nv-codec-headers` you checked out is not compatible with your NVIDIA driver, CUDA Toolkit, or the version of FFmpeg you are building.
-   **Solution**:
    1.  Carefully re-check your [NVIDIA Driver](https://www.nvidia.com/Download/index.aspx) and [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) versions.
    2.  Go back to [Step 3: Install NVIDIA Codec Headers](#step-3-install-nvidia-codec-headers) and ensure you `git checkout` the branch that best matches your environment (e.g., `sdk/12.6`).
    3.  Consult the [Official NVIDIA FFmpeg Guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/index.html) or the `nv-codec-headers` repository to confirm version compatibility.

## References

-   [VMAF on GitHub](https://github.com/Netflix/vmaf)
-   [FFmpeg Official Source](https://github.com/FFmpeg/FFmpeg/tree/release/6.1)
-   [NVIDIA Codec Headers Source](https://github.com/FFmpeg/nv-codec-headers/tree/sdk/12.6)
-   [Official NVIDIA Guide for Compiling FFmpeg](https://docs.nvidia.com/video-technologies/video-codec-sdk/ffmpeg-with-nvidia-gpu/index.html)