## 环境配置

```bash
conda create -n SpatialVid python=3.10
pip install -r requirements/requirements.txt
```

1. scoring 环境

```bash
pip install -r requirements/requirements_scoring.txt
```

2. annotation 环境

```bash
pip install -r requirements/requirements_annotation.txt
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
