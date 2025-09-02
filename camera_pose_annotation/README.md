# Camera Pose Annotation

## Depth Estimation
Use both [Depth-Anything V2](depth_estimation/Depth-Anything) and [UniDepth V2](depth_estimation/UniDepth) to estimate depth maps from images.

Download the pre-trained models from the respective repositories. Skip this step if you already follow the installation instructions in [README](../README.md).
- [Depth-Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Large)
- [UniDepth V2](https://huggingface.co/lpiccinelli/unidepth-v2-vitl14)

To inference depth using Depth-Anything V2, run the following command:

```bash
torchrun --standalone --nproc_per_node ${GPU_NUM} camera_pose_annotation/depth_estimation/Depth-Anything/inference_batch.py \
  ${CSV} \
  --encoder vitl \
  --checkpoints_path checkpoints \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --bs 16 \
  --num_workers ${GPU_NUM}
```

To inference depth using UniDepth V2, run the following command:

```bash
torchrun --standalone --nproc_per_node ${GPU_NUM} camera_pose_annotation/depth_estimation/UniDepth/inference_batch.py \
  ${CSV} \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --checkpoints_path checkpoints \
  --bs 32 \
  --num_workers ${GPU_NUM}
```

## Camera Tracking
Using a DROID-SLAM based method to track camera poses from videos.

To inference a single video, run the following command:

```bash
python camera_pose_annotation/camera_tracking/camera_tracking.py \
  --dir_path ${DIR_PATH} \
  --weights checkpoints/megasam_final.pth \
  --disable_vis
```

To inference videos in batch, run the following command:

```bash
python camera_pose_annotation/camera_tracking/inference_batch.py ${CSV} \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --checkpoints_path checkpoints --gpu_id ${CUDA_VISIBLE_DEVICES} \
  --num_workers $((GPU_NUM * 2))
```

## CVD (Camera View Depth) Optimization
### Optical Flow
Infer optical flow using RAFT model.

Download the [`raft_things.pth`](https://drive.google.com/uc?id=1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM).

To inference a single video, run the following command:

```bash
python camera_pose_annotation/cvd_opt/preprocess/preprocess_flow.py \
  --dir_path ${DIR_PATH} \
  --model checkpoints/raft-things.pth \
  --mixed_precision
```

To inference videos in batch, run the following command:

```bash
python camera_pose_annotation/cvd_opt/preprocess/inference_batch.py ${CSV} \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --checkpoints_path checkpoints --gpu_id ${CUDA_VISIBLE_DEVICES} \
  --num_workers $((GPU_NUM * 2))
```

### Optimization
Using the optical flow to optimize the estimated depth maps.

To inference a single video, run the following command:

```bash
python camera_pose_annotation/cvd_opt/cvd_opt.py \
  --dir_path ${DIR_PATH} \
  --w_grad 2.0 --w_normal 5.0
```

To inference videos in batch, run the following command:

```bash
python camera_pose_annotation/cvd_opt/inference_batch.py ${CSV} \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --gpu_id ${CUDA_VISIBLE_DEVICES} \
  --num_workers $((GPU_NUM * 2))
```

## Dynamic Mask
Given the limitations of MegaSaM in predicting motion probabilities, we opt to enhance its performance using SAM2.

Specifically, an adaptive thresholding mechanism, calibrated to the system’s motion probability distribution, is first employed to generate initial masks. Subsequently, contour detection is performed to mitigate redundant segmentation of overlapping regions; for each identified contour, four evenly spaced anchor points are sampled along its perimeter to serve as dedicated prompts for the SAM2 model.

Download the pre-trained [SAM2 model](https://huggingface.co/facebook/sam2.1-hiera-large).

Run the following command:

```bash
python camera_pose_annotation/dynamic_mask/inference_batch.py ${CSV} \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --checkpoints_path checkpoints --gpu_num ${GPU_NUM} \
  --num_workers $((GPU_NUM * 2))
```
