import os
import json
import concurrent.futures

def convert_keys_to_camel_case(data):
    """
    Convert key names in a dictionary to specified camelCase format.
    Specifically handles certain key transformations as required.
    
    Args:
        data: The input data (dict, list, or other types) to process
        
    Returns:
        Processed data with converted key names where applicable
    """
    if not isinstance(data, dict):
        return data

    new_data = {}
    for key, value in data.items():
        # Key transformation rules
        if key == "categoryTag":
            camel_key = "CategoryTag"
        elif key == "Scene Type":
            camel_key = "sceneType"
        elif key == "First":
            camel_key = "first"
        elif key == "Second":
            camel_key = "second"
        elif key == "Lighting":
            camel_key = "lighting"
        elif key == "Time of Day":
            camel_key = "timeOfDay"
        elif key == "Crowd Density":
            camel_key = "crowdDensity"
        elif key == "Weather":
            camel_key = "weather"
        else:
            camel_key = key  # Keep original key if no transformation rule

        # Recursively process nested structures
        if isinstance(value, dict):
            new_data[camel_key] = convert_keys_to_camel_case(value)
        elif isinstance(value, list):
            # Process each item in the list
            new_data[camel_key] = [convert_keys_to_camel_case(item) for item in value]
        else:
            new_data[camel_key] = value  # Keep original value for non-structured types

    return new_data

def process_single_file(file_path):
    """
    Process a single JSON file to transform its key names according to specified rules.
    
    Args:
        file_path: Path to the JSON file to be processed
        
    Returns:
        Tuple (success_flag, result) where:
        - success_flag is True if processing succeeds, False otherwise
        - result is filename if success, (filename, error_message) if failure
    """
    try:
        # Read and parse JSON data
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle specific transformation for "categoryTag" if present
        if "categoryTag" in data:
            # Convert keys within categoryTag first
            data["categoryTag"] = convert_keys_to_camel_case(data["categoryTag"])
            # Rename categoryTag to CategoryTag
            data["CategoryTag"] = data.pop("categoryTag")
        else:
            # Apply key conversion to entire dataset if no categoryTag
            data = convert_keys_to_camel_case(data)

        # Write modified data back to file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return True, os.path.basename(file_path)

    except Exception as e:
        return False, (os.path.basename(file_path), str(e))

def process_with_timeout(file_path, timeout=2):
    """
    Process a file with timeout protection to prevent hanging on problematic files.
    
    Args:
        file_path: Path to the JSON file to process
        timeout: Maximum allowed time in seconds for processing (default: 2)
        
    Returns:
        Same return format as process_single_file
    """
    # Use thread pool to implement timeout mechanism
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process_single_file, file_path)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return False, (os.path.basename(file_path), "Timeout exceeded")

def process_json_files(directory):
    """
    Main function to process all JSON files in a specified directory.
    Tracks processing results and logs errors.
    
    Args:
        directory: Path to the directory containing JSON files to process
    """
    error_log = []
    processed_count = 0

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            processed_count += 1
            file_path = os.path.join(directory, filename)

            print(f"Processing {filename}...")

            try:
                # Process file with timeout
                success, result = process_with_timeout(file_path, timeout=2)

                if success:
                    print(f"{processed_count} ✅ Processed: {result}")
                else:
                    fname, err = result
                    print(f"{processed_count} ❌ Error processing {fname}: {err}")
                    error_log.append(f"{fname}: {err}")
            except Exception as e:
                print(f"{processed_count} ❌ Unexpected error: {e}")
                error_log.append(f"{filename}: Unexpected error - {e}")

    # Write error log to file
    error_log_path = "error.log"
    with open(error_log_path, "w", encoding="utf-8") as f:
        for line in error_log:
            f.write(line + "\n")

    # Print summary statistics
    print(f"\n✅ Total files processed: {processed_count}")
    print(f"❌ Errors occurred in {len(error_log)} files. See '{error_log_path}' for details.")

# Execute the main function (please modify the directory path as needed)
process_json_files("/share/wjh/opencam/datasets/annotations/captions/group_0")