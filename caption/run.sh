#!/bin/bash
#

# VQA
vqa_prompt_file="./VQA/vqa_prompt2.txt"
vqa_model="gemini-2.0-flash"
vqa_api_key="sk-3hsgJRTTd9492B3947a0T3BlbKFJ2Ca08307C2Ac4721ac3f" 
vqa_num_workers=16 #线程数
vqa_base_domain="https://cn2us02.opapi.win/"
vqa_domain_file="./VQA/vqa_base_domain.txt"
vqa_wait_time=1 #线程发送等待时间

# LLM
llm_prompt_dir="./LLM/prompts2"
llm_num_workers=16 #线程数
llm_wait_time=0.5 #线程发送等待时间

qwen_base_domain="https://dashscope.aliyuncs.com/compatible-mode/"
qwen_api_key_file="./LLM/api_list2.txt"
qwen_model_file="./LLM/model_list2.txt"

for i in {32..35}; do
    group_id="group_${i}"
    # 输出当前执行的group_id
    echo "当前: $group_id"
    python ./VQA/vqa_run.py \
        --group_id $group_id \
        --prompt_file $vqa_prompt_file \
        --model $vqa_model \
        --api_key $vqa_api_key \
        --num_workers $vqa_num_workers \
        --domain_file $vqa_domain_file \
        --wait_time $vqa_wait_time \
      #   --record_time # 可选参数，用于记录请求用时
    
    echo "正在清理$group_id 中VQA的错误数据"
    python ./VQA/clean_vqa.py $group_id

    python ./LLM/llm_run.py \
      --group_id $group_id \
      --prompt_dir $llm_prompt_dir \
      --model_file $qwen_model_file \
      --api_key_file $qwen_api_key_file \
      --num_workers $llm_num_workers \
      --base_domain $qwen_base_domain \
      --wait_time $llm_wait_time \
    #   --record_time # 可选参数，用于记录请求用时

    echo "正在清理$group_id 中LLM的错误数据"
    python ./LLM/clean_llm.py $group_id

    echo "合并$group_id 的数据"
    python ./post_caption/combine.py \
      --group_id $group_id
done