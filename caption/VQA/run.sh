#!/bin/bash

# id_list=("2" "3")
# for id in "${id_list[@]}"; do
#     group_id="group_${id}"

# 获取输入目录路径
# group_id="group_12"
prompt_file="./vqa_prompt2.txt"
model="gemini-2.0-flash"
api_key="sk-3hsgJRTTd9492B3947a0T3BlbKFJ2Ca08307C2Ac4721ac3f" 
num_workers=16 #线程数
base_domain="https://cn2us02.opapi.win/"
# base_domain="https://www.aigptx.top/"
# base_domain="https://c-z0-api-01.hash070.com/"
# base_domain="https://cfwus02.opapi.win"
wait_time=0.5 #线程发送等待时间


# python ./vqa_run.py \
#       --group_id $group_id \
#       --prompt_file $prompt_file \
#       --model $model \
#       --api_key $api_key \
#       --num_workers $num_workers \
#       --base_domain $base_domain \
#       --wait_time $wait_time \
#    #    --record_time # 可选参数，用于记录请求用时

# 按顺序执行group_1, group_2, group_3 一直到 group_15
# 串行化执行
for i in {26..35}; do
    group_id="group_${i}"
    # 输出当前执行的group_id
    echo "当前: $group_id"
    python ./vqa_run.py \
        --group_id $group_id \
        --prompt_file $prompt_file \
        --model $model \
        --api_key $api_key \
        --num_workers $num_workers \
        --base_domain $base_domain \
        --wait_time $wait_time \
      #   --record_time # 可选参数，用于记录请求用时
done