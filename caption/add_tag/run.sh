#!/bin/bash

##############################################################################
# LLM (Large Language Model) Configuration
##############################################################################
# Directory for LLM prompt file
llm_prompt_dir="./tag_prompt.txt"

# Number of worker threads for concurrent processing
llm_num_workers=80

# Waiting time between thread requests (in seconds)
llm_wait_time=0.5


##############################################################################
# Qwen API Configuration
##############################################################################
# Base domain for Qwen API service
qwen_base_domain="https://www.dmxapi.cn/"

# File containing API keys for authentication
qwen_api_key_file="./api.txt"

# File containing list of available models
qwen_model_file="./model_list.txt"


##############################################################################
# Processing Parameters
##############################################################################
# Number of retry attempts for small loop operations
llm_re_try=1

# Start and end IDs for the main processing loop (reserved for future use)
start_id=114
end_id=693


##############################################################################
# Special ID Processing
##############################################################################
# List of IDs that require special handling
special_ids="124 132 404 546"

# Process each special ID individually
for id in $special_ids; do
    echo "Processing data with ID: $id"
    
    # Create group ID based on current ID
    group_id="group_${id}"
    
    # Execute the tag processing Python script with configured parameters
    python ./tag_run.py \
    --group_id $group_id \
    --prompt_dir $llm_prompt_dir \
    --model_file $qwen_model_file \
    --api_key_file $qwen_api_key_file \
    --num_workers $llm_num_workers \
    --base_domain $qwen_base_domain \
    --wait_time $llm_wait_time \
    #   --record_time # Optional parameter to record request execution time
done
