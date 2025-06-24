# Installation

## Getting Started
```bash
git clone https://github.com/FelixYuan-YF/step1_scoring.git
cd step1_scoring
```

## Data Dependencies
Before running the scoring tools, you need to download the following packages:
```bash
# create a new conda environment
conda create -n scoring python=3.10.13
conda activate scoring

# install requirements
pip install -r requirements.txt
```

However, if your videos are in av1 codec instead of h264, you need to install ffmpeg (already in our [requirement script](./requirements.txt)), then run the following to make conda support av1 codec:

```bash
pip uninstall opencv-python
conda install -c conda-forge opencv
```

### Potential Issues
You may encounter some issues with `cpbd` and `mmdet` versions, you can follow the instructions below to fix them.

#### CPBD

You need to go into `path_to_your_env/lib/python3.10/site-packages/cpbd/compute.py` and change the import of `from scipy.ndimage import imread` to `from imageio import imread`.

If you are unsure of your path to the cpbd compute file, simply run our [blur command](./tools/scoring/README.md), wait for the cpbd assertion error on scipy.
The error will contain the exact path to the cpbd compute file.

#### MMDet

You need to go into `path_to_your_env/lib/python3.10/site-packages/mmdet/__init__.py`
and change the assert of `mmcv_version < digit_version(mmcv_maximum_version)` to `mmcv_version <= digit_version(mmcv_maximum_version)`.

If you are unsure of your path to the mmdet init file, simply run our [OCR command](./tools/scoring/README.md), wait for the mmdeet assertion error on mmcv versions.
The error will contain the exact path to the mmdet init file.

#### MMCV
If you encounter the `ModuleNotFoundError: No module named 'mmcv._ext'` error when running OCR, the reason may be that the mmcv version installed by pip is inconsistent with the version we used to build the project. You can try replacing the `path_to_your_env/lib/python3.10/site-packages/mmcv` folder with the [mmcv folder](./package) we provided in our project, and then run OCR again.

If you are unsure of your path to the mmcv folder, simply run our [OCR command](./tools/scoring/README.md), wait for the mmcv assertion error.

## Test
After installation, you can run the following command to test the installation:

```bash
bash run_test.sh
```
