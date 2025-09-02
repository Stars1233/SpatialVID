<h1 align='center'>SpatialVid: A Large Scale Video Dataset with 3D Annotation</h1>
<div align='center'>
    <a href='#' target='_blank'>Jiahao Wang</a>,
    <a href='#' target='_blank'>Yufeng Yuan</a>,
    <a href='#' target='_blank'>Rujie Zheng</a>,
    <a href='#' target='_blank'>Youtian Lin</a>,
    <a href='#' target='_blank'>Yi Zhang</a>,
    <a href='#' target='_blank'>Yajie Bao</a>,
</div>
<div align='center'>
    <a href='#' target='_blank'>Lin-Zhuo Chen</a>,
    <a href='#' target='_blank'>Yanxi Zhou</a>,
    <a href='#' target='_blank'>Xun Cao</a>,
    <a href='#' target='_blank'>Yao Yao</a><sup>†</sup>
</div>
<div align='center'>
    Nanjing University
</div>
<br>
<div align="center">
  <a href="https://nju-pcalab.github.io/projects/openvid/"><img src="https://img.shields.io/static/v1?label=SpatialVid&message=Project&color=purple"></a>  
  <a href="https://arxiv.org/abs/2407.02371"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv"></a>  
  <a href="https://github.com/opencam-vid/SpatialVid"><img src="https://img.shields.io/static/v1?label=Code&message=Github&color=blue&logo=github"></a>  
  <a href="https://huggingface.co/SpatialVid"><img src="https://img.shields.io/static/v1?label=Dataset&message=HuggingFace&color=yellow&logo=huggingface"></a>  
</div>
<p align="center">
  <img src="assets/overview.jpg"  height=400>
</p>

## Introduction

SpatialVid is a large-scale and high-quality video dataset designed for

## 🎉NEWS
+ [2025.09.01] 🔥 Our paper, code and SpatialVid-HQ dataset are released!

## Preparation

### Environment

1. Necessary packages

   ```bash
   git clone --recursive https://github.com/opencam-vid/SpatialVid.git
   cd SpatialVid
   conda create -n SpatialVid python=3.10.13
   conda activate SpatialVid
   pip install -r requirements/requirements.txt
   ```
2. Package needed for scoring

   ```bash
   pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
   pip install -r requirements/requirements_scoring.txt
   ```

   Ignore the warning about `nvidia-nccl-cu12` version, it is not a problem.

   About FFMPEG ？

   Replace the `FFMPEG_PATH` variable in the [`scoring/motion/inference.py`](scoring/motion/inference.py) and [`utils/cut.py`](utils/cut.py) with the actual path to your ffmpeg executable, default is `/usr/local/bin/ffmpeg`.

   [Optional] if your videos are in av1 codec instead of h264, you need to install ffmpeg (already in our requirement script), then run the following to make conda support av1 codec:

   ```bash
   pip uninstall opencv-python
   conda install -c conda-forge opencv
   ```
3. Package needed for annotation

   ```bash
   pip install -r requirements/requirements_annotation.txt
   ```

   Compile the extensions for the camera tracking module:

   ```bash
   cd camera_pose_annotation/base
   python setup.py install
   ```
4. Package needed for caption

   ```bash
   pip install -r requirements/requirements_caption.txt
   ```

### Model Weight

Download the model weights used in our experiments:

```bash
bash scripts/download_checkpoints.sh
```

Or you can manually download the model weights from the following links and place them in the appropriate directories.

| Model               | File Name               | URL                                                                                                             |
| ------------------- | ----------------------- | --------------------------------------------------------------------------------------------------------------- |
| Aesthetic Predictor | aesthetic               | [🔗](https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth) |
| RAFT                | raft-things             | [🔗](https://drive.google.com/uc?id=1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM)                                             |
| Depth Anything      | Depth-Anything-V2-Large | [🔗](https://huggingface.co/depth-anything/Depth-Anything-V2-Large)                                                |
| UniDepth            | unidepth-v2-vitl14      | [🔗](https://huggingface.co/lpiccinelli/unidepth-v2-vitl14)                                                        |
| SAM                 | sam2.1-hiera-large      | [🔗](https://huggingface.co/facebook/sam2.1-hiera-large)                                                           |

## Quick Start

The whole pipeline is illustrated in the figure below:

<p align="center">
  <img src="assets/pipeline.jpg"  height=340>
</p>

1. Scoring

   ```bash
   bash scripts/scoring.sh
   ```

   Inside the [`scoring.sh`](scripts/scoring.sh) script, you need to set the following variables:

   - `ROOT_VIDEO` is the directory containing the input video files.
   - `OUTPUT_DIR` is the directory where the output files will be saved.
2. Annotation

   ```bash
   bash scripts/annotation.sh
   ```

   Inside the [`annotation.sh`](scripts/annotation.sh) script, you need to set the following variables:

   - `CSV` is the CSV file generated by the scoring script, default is `$OUTPUT_DIR/meta/clips_scores.csv`.
   - `OUTPUT_DIR` is the directory where the output files will be saved.
3. Caption

   ```bash
   bash scripts/caption.sh
   ```

## References

Thanks to the developers and contributors of the following open-source repositories, whose invaluable work has greatly inspire our project:

- [Open-Sora](https://github.com/hpcaitech/Open-Sora): An initiative dedicated to efficiently producing high-quality video.
- [MegaSaM](https://github.com/mega-sam/mega-sam): An accurate, fast and robust casual structure and motion from casual dynamic videos.
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2): A model for monocular depth estimation.
- [UniDepthV2](https://github.com/lpiccinelli-eth/UniDepth): A model for universal monocular metric depth estimation.
- [SAM2](https://github.com/facebookresearch/sam2): A model towards solving promptable visual segmentation in images and videos.
- [Viser](https://viser.studio/latest/): A library for interactive 3D visualization in Python.

This project is licensed under Apache License. However, if you use MegaSaM or other components in your work, please follow their license.

## Citation

```bibtex

```
