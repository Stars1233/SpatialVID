#!/bin/bash
CSV=[Replace with the path to the CSV file generated in the scoring step]
OUTPUT_DIR=[Replace with the path to your output directory]

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_NUM=8

measure_time() {
    local step_number=$1
    shift
    local green="\e[32m"
    local red="\e[31m"
    local no_color="\e[0m"
    local yellow="\e[33m"
    
    start_time=$(date +%s)
    echo -e "${green}Step $step_number started at: $(date)${no_color}"

    "$@"

    end_time=$(date +%s)
    echo -e "${red}Step $step_number finished at: $(date)${no_color}"
    echo -e "${yellow}Duration: $((end_time - start_time)) seconds${no_color}"
    echo "---------------------------------------"
}

# 1. Extract frames
measure_time 1 python utils/extract_frames.py ${CSV} --output_folder ${OUTPUT_DIR} \
  --num_workers $((GPU_NUM * 2)) --interval 0.2

# 2.1 Depth Estimation with Depth-Anything
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 2.1 torchrun --standalone --nproc_per_node ${GPU_NUM} camera_pose_annotation/depth_estimation/Depth-Anything/inference_batch.py \
  ${CSV} \
  --encoder vitl \
  --checkpoints_path checkpoints \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --bs 16 \
  --num_workers ${GPU_NUM}

# 2.2 Depth Estimation with UniDepth
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 2.2 torchrun --standalone --nproc_per_node ${GPU_NUM} camera_pose_annotation/depth_estimation/UniDepth/inference_batch.py \
  ${CSV} \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --checkpoints_path checkpoints \
  --bs 32 \
  --num_workers ${GPU_NUM}

# 3. Camera Tracking
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 3 python camera_pose_annotation/camera_tracking/inference_batch.py ${CSV} \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --checkpoints_path checkpoints --gpu_id ${CUDA_VISIBLE_DEVICES} \
  --num_workers $((GPU_NUM * 2))

# 4.1 CVD Optimization Preprocess
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 4.1 python camera_pose_annotation/cvd_opt/preprocess/inference_batch.py ${CSV} \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --checkpoints_path checkpoints --gpu_id ${CUDA_VISIBLE_DEVICES} \
  --num_workers $((GPU_NUM * 2))

# 4.2 CVD Optimization
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 4.2 python camera_pose_annotation/cvd_opt/inference_batch.py ${CSV} \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --gpu_id ${CUDA_VISIBLE_DEVICES} \
  --num_workers $((GPU_NUM * 2))

# 5. Convert the output poses.npy into a c2w matrix
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 5 python utils/c2w.py ${CSV} --OUTPUT_DIR ${OUTPUT_DIR} \
  --num_workers $((GPU_NUM * 2))

# 6. Dynamic Mask Prediction
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 6 python camera_pose_annotation/dynamic_mask/inference_batch.py ${CSV} --OUTPUT_DIR ${OUTPUT_DIR} \
  --checkpoints_path checkpoints --gpu_num ${GPU_NUM} \
  --num_workers $((GPU_NUM * 2))

# 7. Evaluation of the results
measure_time 7 python utils/evaluation.py ${CSV} --OUTPUT_DIR ${OUTPUT_DIR} \
  --gpu_id ${CUDA_VISIBLE_DEVICES} --num_workers $((GPU_NUM * 2)) \
  --output_path ${OUTPUT_DIR}/final_results.csv