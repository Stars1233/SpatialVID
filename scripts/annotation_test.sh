#!/bin/bash
# put each evalset into python scripts (one by one) 
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

CUDA_LIST=0,1,2,3,4,5,6,7
GPU_NUM=8

CSV=test/outputs_scoring/meta/clips_scores.csv
DIR_PATH=test/outputs_annotation

measure_time 1 python utils/extract_frames.py ${CSV} --output_folder ${DIR_PATH} \
  --num_workers $((GPU_NUM * 2)) --interval 0.2

CUDA_VISIBLE_DEVICES=${CUDA_LIST} measure_time 2 torchrun --standalone --nproc_per_node ${GPU_NUM} camera_pose_annotation/depth_estimation/Depth-Anything/inference_batch.py \
  ${CSV} \
  --encoder vitl \
  --checkpoints_path checkpoints \
  --dir_path ${DIR_PATH} \
  --bs 16 \
  --num_workers ${GPU_NUM}

CUDA_VISIBLE_DEVICES=${CUDA_LIST} measure_time 3 torchrun --standalone --nproc_per_node ${GPU_NUM} camera_pose_annotation/depth_estimation/UniDepth/inference_batch.py \
  ${CSV} \
  --dir_path ${DIR_PATH} \
  --checkpoints_path checkpoints \
  --bs 32 \
  --num_workers ${GPU_NUM}

CUDA_VISIBLE_DEVICES=${CUDA_LIST} measure_time 4 python camera_pose_annotation/camera_tracking/inference_batch.py ${CSV} \
  --dir_path ${DIR_PATH} \
  --checkpoints_path checkpoints --gpu_id ${CUDA_LIST} \
  --num_workers $((GPU_NUM * 2))

CUDA_VISIBLE_DEVICES=${CUDA_LIST} measure_time 5 python camera_pose_annotation/cvd_opt/preprocess/inference_batch.py ${CSV} \
  --dir_path ${DIR_PATH} \
  --checkpoints_path checkpoints --gpu_id ${CUDA_LIST} \
  --num_workers $((GPU_NUM * 2))

CUDA_VISIBLE_DEVICES=${CUDA_LIST} measure_time 6 python camera_pose_annotation/cvd_opt/inference_batch.py ${CSV} \
  --dir_path ${DIR_PATH} \
  --gpu_id ${CUDA_LIST} \
  --num_workers $((GPU_NUM * 2))

CUDA_VISIBLE_DEVICES=${CUDA_LIST} measure_time 7 python utils/c2w.py ${CSV} --dir_path ${DIR_PATH} \
  --num_workers $((GPU_NUM * 2))

CUDA_VISIBLE_DEVICES=${CUDA_LIST} measure_time 8 python camera_pose_annotation/dynamic_mask/inference_batch.py ${CSV} --dir_path ${DIR_PATH} \
  --checkpoints_path checkpoints --gpu_num ${GPU_NUM} \
  --num_workers $((GPU_NUM * 2))

measure_time 9 python utils/evaluation.py ${CSV} --dir_path ${DIR_PATH} \
  --gpu_id ${CUDA_LIST} --num_workers $((GPU_NUM * 2)) \
  --output_path ${DIR_PATH}/results.csv
