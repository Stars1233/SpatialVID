#!/bin/bash

##############################################################
# Script Name: run_process_npy.sh
# Description: Invokes process_npy.py to handle CSV and NPY files
# Usage: Directly run this script (parameters are explicitly defined within)
##############################################################

# --------------------------
# Parameter Definitions
# --------------------------
# Define data directory path
DATA_DIR="/home/zrj/project/api_test/pipeline/data"
# Define CSV filename to be processed
CSV_FILENAME="stage1_total_done_sample_200.csv"


# --------------------------
# Pre-execution Checks
# --------------------------
# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist"
    exit 1
fi

# Construct full CSV file path and verify existence
CSV_PATH="$DATA_DIR/$CSV_FILENAME"
if [ ! -f "$CSV_PATH" ]; then
    echo "Error: CSV file '$CSV_PATH' does not exist"
    exit 1
fi

# Define path to Python processing script
SCRIPT_PATH="./process_npy.py"

# Verify Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script '$SCRIPT_PATH' does not exist"
    exit 1
fi


# --------------------------
# Script Execution
# --------------------------
# Execute Python script with defined parameters
echo "Executing: python $SCRIPT_PATH to process $CSV_FILENAME in $DATA_DIR"
python "$SCRIPT_PATH" "$DATA_DIR" "$CSV_FILENAME"


# --------------------------
# Execution Result Check
# --------------------------
# Check if Python script executed successfully
if [ $? -eq 0 ]; then
    echo "Processing completed successfully!"
else
    echo "An error occurred during processing. Please check logs."
    exit 1
fi
