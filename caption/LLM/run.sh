#!/bin/bash

# ==============================================================================
# Configuration Section - Adjust these parameters as needed
# ==============================================================================

# Directory containing prompt files
prompt_dir="./prompts"

# Number of worker threads for parallel processing
num_workers=24

# Wait time (in seconds) between thread submissions to avoid rate limiting
wait_time=0.2

# Number of retries for LLM processing if it fails
llm_re_try=1

# Qwen model API configuration
qwen_base_domain="https://dashscope.aliyuncs.com/compatible-mode/"
qwen_api_key="sk-****"  # Default API key (can be overridden by file)
qwen_api_key_file="./api_list.txt"  # File containing list of API keys
qwen_model_file="./model_list.txt"  # File containing list of models to use

# ==============================================================================
# Processing Section - Main workflow execution
# ==============================================================================

# Process groups from 98 to 112 (inclusive)
for group_number in {98..112}; do
    # Create group identifier
    group_id="group_${group_number}"
    
    # Retry processing up to the specified number of attempts
    for retry_attempt in $(seq 1 $llm_re_try); do
        # Log current processing status
        echo "------------------------------------------------"
        echo "Processing: $group_id (Attempt: $retry_attempt/$llm_re_try)"
        echo "------------------------------------------------"

        # Run LLM processing script with current parameters
        python ./llm_run.py \
            --group_id "$group_id" \
            --prompt_dir "$prompt_dir" \
            --model_file "$qwen_model_file" \
            --api_key_file "$qwen_api_key_file" \
            --num_workers "$num_workers" \
            --base_domain "$qwen_base_domain" \
            --wait_time "$wait_time"
            # Uncomment below to enable request time recording
            # --record_time

        # Run cleanup script for the processed group
        python clean_llm.py "$group_id"
    done 
done

echo "All groups processed successfully."
