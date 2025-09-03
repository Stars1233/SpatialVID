import os
import sys
import csv
import json
import re
import queue
import argparse
from typing import List, Dict, Any, Optional
from multiprocessing import Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Constants for file paths
BASE_TMP_PATH = "/share/wjh/opencam/datasets/annotations/tmp_captions"
BASE_OUTPUT_PATH = "/share/wjh/opencam/datasets/annotations/captions"
BASE_META_INFO_PATH = "/share/wjh/opencam/meta_infos/sam2_infos"
ERROR_LOG_FILE = "combine_error.log"


def parse_text_to_json(text: str) -> Dict[str, Any]:
    """
    Parses text in a specific format into a JSON structure.
    
    This function processes text content with specific labels and converts
    it into a dictionary (JSON structure) with corresponding keys.
    
    Args:
        text: Input text in a specific format
        
    Returns:
        Parsed JSON dictionary with labeled content
    """
    # Define mapping between text labels and JSON keys
    labels = {
        "Camera Motion Caption": "OptCamMotion",
        "Scene Abstract Caption": "SceneSummary",
        "Main Motion Trend Summary": "MotionTrends",
        "Scene Keywords": "SceneTags",
        "Immersive Shot Summary": "ShotImmersion",
        "Qwen model": "LLM"
    }
    
    # Initialize result dictionary with empty values
    result = {key: "" for key in labels.values()}
    current_label = None
    current_content = []
    
    # Process text line by line
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if current line contains any label
        for label, json_key in labels.items():
            if label in line:
                # Find position of first letter after the label
                start_pos = line.find(label) + len(label)
                # Skip non-alphabet characters
                while start_pos < len(line) and not line[start_pos].isalpha():
                    start_pos += 1
                
                content = line[start_pos:].strip()
                
                # Process content if it exists after the label
                if content:
                    if json_key in ["MotionTrends", "SceneTags"]:
                        # Split by commas (both Chinese and English), preserve spaces in phrases
                        items = re.split(r'[，,]\s*', content)
                        result[json_key] = [item.strip() for item in items if item.strip()]
                    else:
                        result[json_key] = content
                    
                    current_label = None
                    current_content = []
                else:
                    # No content after label, continue reading subsequent lines
                    current_label = json_key
                    current_content = []
                
                break
        else:
            # If collecting content for a label
            if current_label:
                # Check if line is not empty
                if line:
                    # Find first letter in line
                    start_pos = 0
                    while start_pos < len(line) and not line[start_pos].isalpha():
                        start_pos += 1
                    
                    if start_pos < len(line):
                        current_content.append(line[start_pos:])
                else:
                    # Empty line indicates end of current label content
                    content = ' '.join(current_content).strip()
                    
                    if current_label in ["MotionTrends", "SceneTags"]:
                        # Split by commas (both Chinese and English), preserve spaces in phrases
                        items = re.split(r'[，,]\s*', content)
                        result[current_label] = [item.strip() for item in items if item.strip()]
                    else:
                        result[current_label] = content
                    
                    current_label = None
                    current_content = []
        
        i += 1
    
    # Handle Qwen model label which might extend to the end
    if current_label == "LLM" and current_content:
        content = ' '.join(current_content).strip()
        result[current_label] = content
    
    return result


def vqa_parse_text_to_json(text: str) -> Dict[str, Any]:
    """
    Parses text containing Camera Motion Caption and Scene Description into JSON format.
    
    Args:
        text: Input text in a specific format
        
    Returns:
        Parsed JSON dictionary with camera motion and scene description
    """
    result = {
        "CamMotion": "",
        "SceneDesc": ""
    }
    
    # Process Camera Motion Caption - from first letter after label to newline
    camera_pattern = r'Camera Motion Caption:\s*(\w[\s\S]*?)(?=\n|$)'
    camera_match = re.search(camera_pattern, text)
    if camera_match:
        result["CamMotion"] = camera_match.group(1).strip()
    
    # Process Scene Description - from first letter after label to end of text
    scene_pattern = r'Scene Description:\s*(\w[\s\S]*)$'
    scene_match = re.search(scene_pattern, text, re.DOTALL)
    if scene_match:
        result["SceneDesc"] = scene_match.group(1).strip()
    
    return result


def process_single_row(group_id: str, scene_id: str, num_workers: int = 4) -> None:
    """
    Processes VQA and LLM captions for a single scene and merges them into one JSON file.
    
    Args:
        group_id: Batch ID
        scene_id: Scene ID
        num_workers: Number of worker threads
    """
    # Define file paths
    vqa_path = os.path.join(BASE_TMP_PATH, f"{group_id}/vqa_tmp")
    llm_path = os.path.join(BASE_TMP_PATH, f"{group_id}/llm_tmp")
    output_path = os.path.join(BASE_OUTPUT_PATH, f"{group_id}")
    
    # Build full file paths
    vqa_file = os.path.join(vqa_path, f"{scene_id}.txt")
    llm_file = os.path.join(llm_path, f"{scene_id}.txt")
    output_file = os.path.join(output_path, f"{scene_id}.json")
    
    # Skip if output file already exists
    if os.path.exists(output_file):
        return
    
    # Check if input files exist
    if not os.path.isfile(vqa_file):
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{group_id}/{scene_id}: no VQA file\n")
        print(f"VQA file not found: {vqa_file}")
        return
    
    if not os.path.isfile(llm_file):
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{group_id}/{scene_id}: no LLM file\n")
        print(f"LLM file not found: {llm_file}")
        return
    
    try:
        # Read VQA file content
        with open(vqa_file, 'r', encoding='utf-8') as f:
            vqa_text = f.read()
        
        # Read LLM file content
        with open(llm_file, 'r', encoding='utf-8') as f:
            llm_text = f.read()
        
        # Parse text content to JSON
        vqa_json = vqa_parse_text_to_json(vqa_text)
        llm_json = parse_text_to_json(llm_text)
        
        # Merge JSON objects
        combined_json = {**vqa_json, **llm_json}
        
        # Save merged JSON to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_json, f, ensure_ascii=False, indent=2)
        
        print(f"JSON data saved to {output_file}")
    
    except Exception as e:
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{group_id}/{scene_id}\n")
        print(f"Error processing {scene_id}: {e}")


def process(group_id: str, num_workers: int = 32) -> None:
    """
    Processes all scenes in the specified batch.
    
    Args:
        group_id: Batch ID
        num_workers: Number of worker threads
    """
    # Define CSV file path
    csv_path = os.path.join(BASE_META_INFO_PATH, f"{group_id}.csv")
    
    # Check if CSV file exists
    if not os.path.isfile(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    # Ensure output directory exists
    output_path = os.path.join(BASE_OUTPUT_PATH, f"{group_id}")
    os.makedirs(output_path, exist_ok=True)
    
    # Read CSV file
    rows = []
    try:
        with open(csv_path, "r", encoding="utf-8") as fin:
            reader = csv.reader(fin)
            next(reader)  # Skip header row
            rows = list(reader)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Column index representing scene_id
    subdir_col_idx = 1
    
    # Use multiprocessing manager for thread-safe queue
    manager = Manager()
    task_queue = manager.Queue()
    
    # Add tasks to queue
    for idx, row in enumerate(rows):
        scene_id = row[subdir_col_idx]
        task_queue.put((idx, scene_id))
    
    # Define worker function for processing tasks
    def worker(task_queue, pbar):
        while True:
            try:
                idx, scene_id = task_queue.get(timeout=1)
            except queue.Empty:
                break
            
            process_single_row(group_id, scene_id, num_workers)
            task_queue.task_done()
            pbar.update(1)
    
    # Start multi-threaded processing with progress bar
    with tqdm(total=len(rows), desc="Processing progress") as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _ in range(num_workers):
                futures.append(executor.submit(worker, task_queue, pbar))
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                future.result()
    
    print(f"Batch {group_id} processing completed, total {len(rows)} scenes processed")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Merge VQA and LLM caption data')
    parser.add_argument('--group_id', type=str, required=True, help='Batch ID')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of worker threads')
    return parser.parse_args()


def main() -> None:
    """Main function that processes command line arguments and starts processing."""
    args = parse_args()
    process(args.group_id, args.num_workers)


if __name__ == "__main__":
    main()
