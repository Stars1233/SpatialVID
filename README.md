```bash
git clone --recursive https://github.com/opencam-vid/SpatialVid.git
cd SpatialVid
```

## 环境配置

```bash
conda create -n SpatialVid python=3.10.13
pip install -r requirements/requirements.txt
```

1. scoring 环境

```bash
pip install -r requirements/requirements_scoring.txt
```

You need to go into `path_to_your_env/lib/python3.10/site-packages/cpbd/compute.py` and change the import of `from scipy.ndimage import imread` to `from imageio import imread`.

However, if your videos are in av1 codec instead of h264, you need to install ffmpeg (already in our requirement script), then run the following to make conda support av1 codec:

```bash
pip uninstall opencv-python
conda install -c conda-forge opencv
```

2. annotation 环境

```bash
pip install -r requirements/requirements_annotation.txt
```

Compile the extensions for the camera tracking module:

```bash
cd camera_pose_annotation/base
python setup.py install
```

## checkpoints 下载

```bash
bash scripts/download_checkpoints.sh
```

## 测试

```bash
bash scripts/scoring_test.sh
bash scripts/annotation_test.sh
```
