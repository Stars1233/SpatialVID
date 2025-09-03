import os
import sys
import csv
import time
import random
import queue
from argparse import ArgumentParser
from multiprocessing import Manager
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent
# Add current directory to system path for module import
sys.path.append(os.path.dirname(__file__))
from utils import api_call


def process_single_row(group_id, scene_id, prompt_dir, model, api_key, vqa_caption, 
                      motion_intensity, base_domain="https://cn2us02.opapi.win/"):
    """
    Process a single row of data by calling API and saving the result
    
    Args:
        group_id (str): Identifier for the group
        scene_id (str): Identifier for the scene
        prompt_dir (str): Directory containing prompt files
        model (str): Name of the model to use
        api_key (str): API key for authentication
        vqa_caption (str): Caption from VQA processing
        motion_intensity (str): Motion intensity value
        base_domain (str): Base domain for API calls
        
    Returns:
        None
    """
    # Temporary path for storing captions
    tmp_path = "/share/wjh/opencam/datasets/annotations/tmp_captions"
    if not os.path.isdir(tmp_path):
        print(f"Subdirectory not found: {tmp_path}")
        return
        
    # Create LLM output directory if it doesn't exist
    llm_path = os.path.join(tmp_path, f"{group_id}/llm_tmp")
    if not os.path.isdir(llm_path):
        os.makedirs(llm_path, exist_ok=True)
        print(f"Created directory: {llm_path}")
    
    # Skip processing if file already exists
    save_file = os.path.join(llm_path, f"{scene_id}.txt")
    if os.path.exists(save_file):
        return

    # Call API with retry mechanism
    llm_caption = api_call(group_id, scene_id, prompt_dir, model, api_key, 
                          vqa_caption, motion_intensity, base_domain)
    
    # Retry up to 15 times if API call fails
    retry_count = 15
    while llm_caption is None and retry_count > 0:
        llm_caption = api_call(group_id, scene_id, prompt_dir, model, api_key, 
                              vqa_caption, motion_intensity, base_domain)
        time.sleep(2)
        retry_count -= 1
    
    # Handle API failure after retries
    if llm_caption is None:
        error_log_path = "llm_error.log"
        with open(error_log_path, 'a', encoding='utf-8') as f:
            # Check if error already logged
            with open(error_log_path, 'r', encoding='utf-8') as rf:
                if f"{group_id}/{scene_id}" not in rf.read():
                    f.write(f"{group_id}/{scene_id}\n")
        print(f"API request error or no result obtained: {scene_id}")
        return

    # Save the result with model information
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write(llm_caption + f"\n\n6. Qwen model: \n{model}")
    return


def process(group_id, prompt_dir, model_file, api_key_file, num_workers=4, 
           wait_time=0.8, base_domain="https://cn2us02.opapi.win/", record_time=False):
    """
    Main processing function that handles multiple rows using parallel workers
    
    Args:
        group_id (str): Identifier for the group
        prompt_dir (str): Directory containing prompt files
        model_file (str): Path to file containing model names
        api_key_file (str): Path to file containing API keys
        num_workers (int): Number of worker threads
        wait_time (float): Time to wait between requests
        base_domain (str): Base domain for API calls
        record_time (bool): Whether to record processing time
        
    Returns:
        None
    """
    # Read models and API keys from files
    with open(model_file, "r") as mf:
        models = [line.strip() for line in mf.readlines()]
    
    with open(api_key_file, "r") as af:
        api_keys = [line.strip() for line in af.readlines()]

    # Validate temporary directory exists
    tmp_path = "/share/wjh/opencam/datasets/annotations/tmp_captions"
    if not os.path.isdir(tmp_path):
        print(f"Subdirectory not found: {tmp_path}")
        return
        
    # Create LLM directory if it doesn't exist
    llm_path = os.path.join(tmp_path, f"{group_id}/llm_tmp")
    if not os.path.isdir(llm_path):
        os.makedirs(llm_path, exist_ok=True)
        print(f"Created directory: {llm_path}")

    # Validate VQA directory exists
    vqa_path = os.path.join(tmp_path, f"{group_id}", "vqa_tmp")
    if not os.path.isdir(vqa_path):
        print(f"Subdirectory not found: {vqa_path}")
        return

    # Read CSV file containing scene information
    base_path = "/share/wjh/opencam/meta_infos/sam2_infos"
    csv_path = os.path.join(base_path, f"{group_id}.csv")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader)  # Skip header row
        rows = list(reader)

    # Column indices for data extraction
    subdir_col_idx = 1  # Index of column containing directory names
    motion_intensity_col_idx = 21  # Index of column containing motion intensity

    def worker(task_queue, pbar):
        """
        Worker function to process tasks from the queue
        
        Args:
            task_queue: Queue containing tasks to process
            pbar: Progress bar object for tracking progress
        """
        while True:
            try:
                idx_row = task_queue.get(timeout=1)
            except queue.Empty:
                break
                
            # Add delay to prevent overwhelming API
            time.sleep(wait_time)

            idx, row = idx_row
            scene_id = row[subdir_col_idx]
            motion_intensity = row[motion_intensity_col_idx]

            # Check if VQA file exists
            vqa_txt_path = os.path.join(vqa_path, f"{scene_id}.txt")
            if not os.path.isfile(vqa_txt_path):
                error_log_path = "llm_error.log"
                # Check if error already logged
                with open(error_log_path, 'r', encoding='utf-8') as f:
                    if f"{group_id}/{scene_id}: no vqa_file\n" not in f.read():
                        with open(error_log_path, 'a', encoding='utf-8') as af:
                            af.write(f"{group_id}/{scene_id}: no vqa_file\n")
                print(f"File not found: {vqa_txt_path}")
                continue
                
            # Read VQA caption
            with open(vqa_txt_path, "r") as f:
                vqa_caption = f.read()
            
            # Add randomness to balance model and key usage
            model_idx = random.randint(0, len(models) - 1)
            key_idx = random.randint(0, len(api_keys) - 1)
            model = models[model_idx]
            api_key = api_keys[key_idx]

            # Process the single row
            process_single_row(group_id, scene_id, prompt_dir, model, api_key, 
                              vqa_caption, motion_intensity, base_domain)
            
            # Update progress
            task_queue.task_done()
            pbar.update(1)
            

    # Initialize task queue with all rows
    manager = Manager()
    task_queue = manager.Queue()
    for idx, row in enumerate(rows):
        task_queue.put((idx, row))

    # Start processing with progress bar
    start_time = time.time() if record_time else None
    with tqdm(total=len(rows), desc=f"({group_id})LLM Finished") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Start worker threads
            futures = [executor.submit(worker, task_queue, pbar) 
                      for _ in range(num_workers)]
            
            # Wait for all workers to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()
    
    # Log processing time if enabled
    if record_time:
        total_time = time.time() - start_time
        if len(rows) > 0:
            print(f"Total API processing time: {total_time:.2f}s, "
                  f"Average time per subdirectory: {total_time/len(rows):.2f}s")
        else:
            print("No valid subdirectories, cannot calculate processing time.")


def parse_args():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments object
    """
    parser = ArgumentParser(description='VQA Processing Program')
    parser.add_argument('--group_id', type=str, help='Batch identifier')
    parser.add_argument('--prompt_dir', type=str, 
                        default=os.path.join(os.path.dirname(__file__), "vqa_prompt.txt"),
                        help='Path to prompt file')
    parser.add_argument('--model_file', type=str, default="model_list.txt",
                        help='Path to model list file')
    parser.add_argument('--api_key_file', type=str, default="api_list.txt",
                        help='Path to API key file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads')
    parser.add_argument('--wait_time', type=float, default=0.8,
                        help='Time between requests in seconds')
    parser.add_argument('--base_domain', type=str, default="https://cn2us02.opapi.win/",
                        help='API base domain')
    parser.add_argument('--record_time', action='store_true',
                        help='Whether to record processing time')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(
        group_id=args.group_id,
        prompt_dir=args.prompt_dir,
        model_file=args.model_file,
        api_key_file=args.api_key_file,
        num_workers=args.num_workers,
        wait_time=args.wait_time,
        base_domain=args.base_domain,
        record_time=args.record_time
    )
