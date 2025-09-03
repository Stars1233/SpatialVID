import os
import sys
import time
import json
import random
import queue
import argparse
from tqdm import tqdm
from multiprocessing import Manager
import concurrent.futures

# Add current directory to system path for module import
sys.path.append(os.path.dirname(__file__))
from utils import api_call


def process_single_row(save_file, prompt_file, model, api_key, base_domain="https://cn2us02.opapi.win/"):
    """
    Process a single JSON file to add category tags via API call
    
    Args:
        save_file (str): Path to the JSON file to process
        prompt_file (str): Path to the prompt file for API call
        model (str): Model name to use for API call
        api_key (str): API key for authentication
        base_domain (str): Base domain for the API endpoint
    """
    # Check if CategoryTag field already exists
    try:
        with open(save_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        # Skip if CategoryTag already exists
        if "CategoryTag" in original_data:
            return
    except (FileNotFoundError, json.JSONDecodeError):
        return

    # Call API to get category tags with retry mechanism
    tag_caption = api_call(save_file, prompt_file, model, api_key, base_domain)
    retry_count = 15
    while tag_caption is None and retry_count > 0:
        tag_caption = api_call(save_file, prompt_file, model, api_key, base_domain)
        time.sleep(2)
        retry_count -= 1

    # Handle API call failure
    if tag_caption is None:
        error_log_path = "add_tag_error.log"
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{save_file}\n")
        print(f"API request error or no result obtained: {save_file}")
        return

    # Parse and add category tags to the JSON file
    category_tag = parse_category_tags(tag_caption)
    
    try:
        with open(save_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return
    
    # Merge new data with existing data
    original_data["CategoryTag"] = category_tag
    
    # Overwrite file with updated data
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, ensure_ascii=False, indent=2)
    
    return


def parse_category_tags(tag_caption):
    """
    Parse API response to structured category data using camelCase naming convention
    
    Args:
        tag_caption (str): Raw response from API containing category information
        
    Returns:
        dict: Structured category data with standardized keys
    """
    lines = [line.strip() for line in tag_caption.strip().split('\n') if line.strip()]
    
    # Initialize category data with default values
    category_data = {
        "sceneType": {
            "first": "Unknown",
            "second": "Unknown"
        },
        "lighting": "Unknown",
        "timeOfDay": "Unknown",
        "weather": "Unknown",
        "crowdDensity": "Unknown"
    }
    
    # Parse each line to extract category information
    for line in lines:
        line_lower = line.lower()
        if line_lower.startswith("primary scene type:"):
            category_data["sceneType"]["first"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("secondary scene type:"):
            category_data["sceneType"]["second"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("lighting:"):
            category_data["lighting"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("time of day:"):
            category_data["timeOfDay"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("weather:"):
            category_data["weather"] = line.split(":", 1)[1].strip()
        elif line_lower.startswith("crowd density:"):
            category_data["crowdDensity"] = line.split(":", 1)[1].strip()
    
    return category_data


def process(group_id, prompt_file, model_file, api_key_file, num_workers=4, 
            wait_time=0.8, base_domain="https://cn2us02.opapi.win/", record_time=False):
    """
    Process a group of JSON files using multiple threads to add category tags
    
    Args:
        group_id (str): Identifier for the group of files to process
        prompt_file (str): Path to the prompt file
        model_file (str): Path to file containing list of models
        api_key_file (str): Path to file containing list of API keys
        num_workers (int): Number of worker threads
        wait_time (float): Time to wait between API calls in seconds
        base_domain (str): Base domain for API endpoint
        record_time (bool): Whether to record and display processing time
    """
    # Load models and API keys from files
    with open(model_file, "r") as mf:
        models = [line.strip() for line in mf.readlines()]
    with open(api_key_file, "r") as af:
        api_keys = [line.strip() for line in af.readlines()]

    # Set up directory path for JSON files
    json_dir = f"/share/wjh/opencam/datasets/annotations/captions/{group_id}"
    if not os.path.isdir(json_dir):
        print(f"JSON file directory not found: {json_dir}")
        return
    
    # Get list of all JSON files in directory
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    
    if not json_files:
        print(f"No JSON files found in directory {json_dir}")
        return

    # Worker function for processing files in parallel
    def worker(task_queue, pbar):
        while True:
            try:
                save_file = task_queue.get(timeout=1)
            except queue.Empty:
                break
                
            time.sleep(wait_time)

            try:
                # Add randomness to balance model and API key usage
                model_idx = random.randint(0, len(models) - 1)
                key_idx = random.randint(0, len(api_keys) - 1)
                
                process_single_row(
                    save_file=save_file,
                    prompt_file=prompt_file,
                    model=models[model_idx],
                    api_key=api_keys[key_idx],
                    base_domain=base_domain
                )
            except Exception as e:
                print(f"Error processing file {save_file}: {str(e)}")
            finally:
                task_queue.task_done()
                pbar.update(1)

    # Initialize task queue and add all files to process
    manager = Manager()
    task_queue = manager.Queue()
    
    for json_file in json_files:
        save_file = os.path.join(json_dir, json_file)
        task_queue.put(save_file)

    # Start processing with progress bar
    start_time = time.time() if record_time else None
    with tqdm(total=len(json_files), desc=f"({group_id}) Tags Completed") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Start worker threads
            futures = [executor.submit(worker, task_queue, pbar) for _ in range(num_workers)]
            
            # Wait for all workers to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()
    
    # Display time statistics if enabled
    if record_time:
        total_time = time.time() - start_time
        if len(json_files) > 0:
            print(f"Total API time: {total_time:.2f}s, Average per JSON file: {total_time/len(json_files):.2f}s")
        else:
            print("No valid JSON files to process, cannot calculate time statistics.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Category Tag Processing Program')
    parser.add_argument('--group_id', type=str, help='Batch sequence number')
    parser.add_argument('--prompt_dir', type=str, 
                        default=os.path.join(os.path.dirname(__file__), "vqa_prompt.txt"),
                        help='Path to prompt file')
    parser.add_argument('--model_file', type=str, default="model_list.txt",
                        help='Path to model list file')
    parser.add_argument('--api_key_file', type=str, default="api_list.txt",
                        help='Path to API key list file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads')
    parser.add_argument('--wait_time', type=float, default=0.8,
                        help='Time interval between requests per thread (seconds)')
    parser.add_argument('--base_domain', type=str, default="https://cn2us02.opapi.win/",
                        help='API base domain')
    parser.add_argument('--record_time', action='store_true',
                        help='Whether to record processing time')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(
        group_id=args.group_id,
        prompt_file=args.prompt_dir,
        model_file=args.model_file,
        api_key_file=args.api_key_file,
        num_workers=args.num_workers,
        wait_time=args.wait_time,
        base_domain=args.base_domain,
        record_time=args.record_time
    )
