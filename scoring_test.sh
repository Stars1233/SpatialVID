ROOT_VIDEO="test/video"
ROOT_CLIPS="test/outputs_scoring/clip"
ROOT_META="test/outputs_scoring/meta"
ROOT_FIG="test/outputs_scoring/fig"
ROOT_LOG="test/outputs_scoring/log"
ROOT_TEMP="test/outputs_scoring/temp"

for dir in $ROOT_VIDEO $ROOT_CLIPS $ROOT_META $ROOT_FIG $ROOT_LOG $ROOT_TEMP; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
    fi
done

GPU_NUM=1
NUM_WORKERS=$((GPU_NUM * 2))
select_detector=7

CUDA_VISIBLE_DEVICES="0"

measure_time() {
  local step_number=$1
  shift
  local green="\e[32m"
  local no_color="\e[0m"
  local yellow="\e[33m"

  start_time=$(date +%s)
  echo -e "${green}Step $step_number started at: $(date)${no_color}"

  # 启动任务，将标准输入重定向到 /dev/null
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES "$@" < /dev/null &
  pid=$!

  # 等待任务结束
  wait $pid

  end_time=$(date +%s)
  echo -e "${green}Step $step_number finished at: $(date)${no_color}"
  echo -e "${yellow}Duration: $((end_time - start_time)) seconds${no_color}"
  echo "---------------------------------------"
}

# 1.1 Create a meta file from a video folder. This should output ${ROOT_META}/meta.csv
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 1.1 python utils/convert.py ${ROOT_VIDEO} --output ${ROOT_META}/meta.csv &> ${ROOT_LOG}/convert.txt

# # 1.2 Get video information and remove broken videos. This should output ${ROOT_META}/meta_info_fmin${fmin_1}.csv
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 1.2 python utils/get_info.py ${ROOT_META}/meta.csv \
#   --num_workers 16 > ${ROOT_LOG}/info.txt

# # 2.1 Detect scenes. This should output ${ROOT_META}/meta_info_fmin${fmin_1}_timestamp.csv
# # Also, you can set the params like "--start-remove-sec 0.5 --end-remove-sec 0.5"
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 2.1 python utils/scene_detect.py ${ROOT_META}/meta_info.csv \
#   --num_workers 64 \
#   --detect_id $select_detector \
#   --frame_skip 2\
#   --start_remove_sec 0.3 --end_remove_sec 0.3 \
#   --min_seconds 2 --max_seconds 15 > ${ROOT_LOG}/scene_detect.txt

# # 2.2 Get clips. This should output ${ROOT_META}/clips_info.csv
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 2.2 python utils/get_clip.py ${ROOT_META}/meta_info_timestamp.csv \
#   --csv_save_dir ${ROOT_META} --num_workers $((GPU_NUM * 4)) > ${ROOT_LOG}/get_clip.txt

# # 2.3 Extract frames for scoring.
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 2.3 python utils/extract_frames.py ${ROOT_META}/clips_info.csv \
#   --output_folder ${ROOT_FIG} --num_workers 64 --target_size "640*360" > ${ROOT_LOG}/extract_frames.txt

# 3.1 Predict aesthetic scores. This should output ${ROOT_META}/clips_info_aes.csv
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 3.1 torchrun --nproc_per_node ${GPU_NUM} scoring/aesthetic/inference.py \
  ${ROOT_META}/clips_info.csv \
  --bs 16 \
  --num_workers ${NUM_WORKERS} \
  --fig_load_dir ${ROOT_FIG} > ${ROOT_LOG}/aes.txt

# 3.2 Predict luminance scores. This should output ${ROOT_META}/clips_info_lum.csv
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 3.2 torchrun --nproc_per_node ${GPU_NUM} scoring/luminance/inference.py \
  ${ROOT_META}/clips_info.csv \
  --bs 16 \
  --num_workers ${NUM_WORKERS} \
  --fig_load_dir ${ROOT_FIG} > ${ROOT_LOG}/lum.txt

# # 3.3 Predict blur scores. This should output ${ROOT_META}/clips_info_blur.csv
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 3.3 torchrun --nproc_per_node ${GPU_NUM} scoring/blur/inference.py \
#   ${ROOT_META}/clips_info.csv \
#   --bs 1 \
#   --num_workers ${NUM_WORKERS} > ${ROOT_LOG}/blur.txt

# 3.4 get optical flow. This should output ${ROOT_META}/clips_info_flow.csv
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 3.4 torchrun --standalone --nproc_per_node ${GPU_NUM} scoring.optical_flow.inference.py \
#   ${ROOT_META}/clips_info.csv \
#   --bs 8 \
#   --num_workers ${NUM_WORKERS} \
#   --fig_load_dir ${ROOT_FIG} \
#   --use_cudnn > ${ROOT_LOG}/flow.txt

# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 3.4 python scoring/motion/inference.py ${ROOT_META}/clips_info.csv \
#   --temp_save_dir ${ROOT_TEMP} \
#   --num_workers $((GPU_NUM * 4)) \
#   --gpu_num ${GPU_NUM} > ${ROOT_LOG}/motion.txt
  
# 3.5 get text by OCR using mmocr's DBNet, this should output ${ROOT_META}/clips_info_ocr.csv
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 3.5 torchrun --standalone --nproc_per_node ${GPU_NUM} -m tools.scoring.ocr.inference \
#   ${ROOT_META}/clips_info.csv \
#   --bs 8 \
#   --num_workers ${NUM_WORKERS} \
#   --fig_load_dir ${ROOT_FIG} > ${ROOT_LOG}/ocr.txt

# # 4.1 camera motion by Gunnar Farneback method, this should output ${ROOT_META}/clips_info_cmotion.csv
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 4.1 python -m tools.caption.camera_motion_detect ${ROOT_META}/clips_info.csv \
#   --num_workers ${NUM_WORKERS} > ${ROOT_LOG}/cmotion.txt

# 5 merge all the scores. This should output ${ROOT_META}/clips_with_score.csv
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 5 python utils/merge_tables.py ${ROOT_META} --output ${ROOT_META}/clips_scores.csv > ${ROOT_LOG}/merge.txt

# # 6 Plot the scores
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 6 python utils/plot_score.py ${ROOT_META}/clips_scores.csv \
#   --num_workers 64 --fig_save_dir ${ROOT_FIG} > ${ROOT_LOG}/plot_score.txt

# 7 Filter the clips.
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m tools.utils.filter ${ROOT_META}/clips_scores.csv \
#   --output ${ROOT_META}/clips_filtered.csv \
#   --num_workers ${NUM_WORKERS} \
#   --aesmin 3 --lummin 5 --flowmin 5 --ocrmax 0 > ${ROOT_LOG}/filter.txt

# 8 Cut the clips.
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES measure_time 8 python -m tools.utils.cut ${ROOT_META}/clips_filtered.csv \
#   --ffmpeg_path ffmpeg \
#   --video_save_dir ${ROOT_CLIPS} --csv_save_dir ${ROOT_META} \
#   --num_workers $((GPU_NUM * 4)) --gpu_num $GPU_NUM &> ${ROOT_LOG}/cut.txt