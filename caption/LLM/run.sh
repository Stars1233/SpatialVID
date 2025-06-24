#!/bin/bash

# 获取输入目录路径
# 设置id列表，分别有1，2，3，4，5，6，7，8，9，10
# id_list=("1" "2" "3")
# for id in "${id_list[@]}"; do
#     group_id="group_${id}"

group_id="group_1"
prompt_dir="./prompts2"
num_workers=32 #线程数
wait_time=0.5 #线程发送等待时间

qwen_base_domain="https://dashscope.aliyuncs.com/compatible-mode/"
qwen_api_key="sk-795881d0b2014166aa2c6407da9015c1"
qwen_model="qwen2.5-72b-instruct"
qwen_api_key_file="./api_list2.txt"
qwen_model_file="./model_list2.txt"

# python ./llm_run.py \
#        --group_id $group_id \
#        --prompt_dir $prompt_dir \
#        --model_file $qwen_model_file \
#        --api_key_file $qwen_api_key_file \
#        --num_workers $num_workers \
#        --base_domain $qwen_base_domain \
#        --wait_time $wait_time \
    #    --record_time # 可选参数，用于记录请求用时

for i in {22..23}; do
    group_id="group_${i}"
    # 输出当前执行的group_id
    echo "当前: $group_id"
    python ./llm_run.py \
       --group_id $group_id \
       --prompt_dir $prompt_dir \
       --model_file $qwen_model_file \
       --api_key_file $qwen_api_key_file \
       --num_workers $num_workers \
       --base_domain $qwen_base_domain \
       --wait_time $wait_time \
      #   --record_time # 可选参数，用于记录请求用时
done