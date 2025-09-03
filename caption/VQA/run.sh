#!/bin/bash

# ==============================================================================
# Batch VQA Processing and Cleaning Script
# Purpose: Runs VQA inference for multiple groups with retries and cleans up error data
# ==============================================================================

# ==============================================================================
# Configuration Section - VQA Parameters
# ==============================================================================

# Path to VQA prompt template file
vqa_prompt_file="./vqa_prompt.txt"

# Model to use for VQA inference
vqa_model="gemini-2.0-flash"

# API authentication
vqa_api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # API key for authentication
vqa_domain_file="./domain.txt"                   # File containing API domain information

# Parallel processing configuration
vqa_num_workers=48                               # Number of parallel worker threads
vqa_wait_time=0.1                                # Delay (in seconds) between thread requests (prevents rate limiting)

# Processing range and retries
vqa_re_try=2                                     # Number of retry attempts per group if processing fails
start_id=450                                     # Starting group ID (inclusive)
end_id=500                                       # Ending group ID (inclusive)

# ==============================================================================
# Main Processing Workflow
# ==============================================================================

# Process each group in the specified ID range
for group_number in $(seq $start_id $end_id); do
    # Generate group identifier (e.g., "group_450")
    group_id="group_${group_number}"
    
    # Display current group being processed
    echo "=========================================="
    echo "Starting processing for: $group_id"
    echo "=========================================="

    # Retry processing up to the configured number of attempts
    for retry_attempt in $(seq 1 $vqa_re_try); do
        echo "--- Attempt $retry_attempt/$vqa_re_try for $group_id ---"
        
        # Run VQA inference script with current parameters
        python ./vqa_run.py \
            --group_id "$group_id" \
            --prompt_file "$vqa_prompt_file" \
            --model "$vqa_model" \
            --api_key "$vqa_api_key" \
            --num_workers "$vqa_num_workers" \
            --domain_file "$vqa_domain_file" \
            --wait_time "$vqa_wait_time"
            # Uncomment the line below to enable request time recording
            # --record_time
        
        # Clean up error data for the current group after inference
        echo "Cleaning VQA error data for $group_id (Attempt $retry_attempt)"
        python ./clean_vqa.py "$group_id"
    done
done

echo "=========================================="
echo "All groups (ID $start_id to $end_id) processed successfully"
echo "=========================================="
