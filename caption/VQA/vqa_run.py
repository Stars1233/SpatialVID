import os
import sys
import csv
import concurrent.futures
from multiprocessing import Manager
import queue
from tqdm import tqdm
import argparse
import time

sys.path.append(os.path.dirname(__file__))
from utils import api_call

def process_single_row(group_id, scene_id, video_path, num_frames, fps, prompt_text, model, api_key, domain_file):
    """
    Process a single row: call the VQA API and save the result for one scene.
    Handles retries and error logging.
    """
    tmp_path = "/share/wjh/opencam/datasets/annotations/tmp_captions"
    VQA_path = os.path.join(tmp_path, f"{group_id}/vqa_tmp")
    if not os.path.isdir(VQA_path):
        os.makedirs(VQA_path, exist_ok=True)
        print(f"Created directory: {VQA_path}")
    save_file = os.path.join(VQA_path, f"{scene_id}.txt")
    if os.path.exists(save_file):
        # Skip if already exists
        return
    # Read all available API domains
    with open(domain_file, "r") as df:
        domains = [line.strip() for line in df.readlines()]
    base_domain = domains[0]
    domain_num = len(domains)
    # Call API
    vqa_caption = api_call(video_path, fps, num_frames, prompt_text, model, api_key, base_domain)
    cnt = 15
    while vqa_caption is None and cnt > 0:
        cnt -= 1
        num = 15 - cnt
        print(f"{group_id}/{scene_id} retry {num}")
        domain_idx = cnt % domain_num
        base_domain = domains[domain_idx]
        vqa_caption = api_call(video_path, fps, num_frames, prompt_text, model, api_key, base_domain)
        time.sleep(2)
    if vqa_caption is None:
        # Log error if still failed
        error_log_path = "vqa_error.log"
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{group_id}/{scene_id}\n")
        print(f"Empty result: {group_id}/{scene_id}")
        return
    # Save result
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write(vqa_caption)
    return

def process(group_id, prompt_file, model, api_key, domain_file, num_workers=4, wait_time = 0.8, record_time=False):
    """
    Batch process all scenes in a group: call VQA API for each row in the CSV.
    Uses a thread pool for concurrency and supports timing.
    """
    import time
    # Read prompt file
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    # Locate CSV file for the group
    base_path = "/share/wjh/opencam/meta_infos/sam2_infos"
    csv_path = os.path.join(base_path, f"{group_id}.csv")
    # Read all rows
    with open(csv_path, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        rows = list(reader)
    # Column indices
    subdir_col_idx = 1
    video_path_col_idx = 0
    num_frames_idx = 2
    fps_idx = 6
    manager = Manager()
    total_time = 0

    def worker(task_queue, pbar):
        while True:
            try:
                idx, row = task_queue.get(timeout=1)
                scene_id = row[subdir_col_idx]
                video_path = row[video_path_col_idx]
                num_frames = int(row[num_frames_idx])
                fps = float(row[fps_idx])
            except queue.Empty:
                break
            time.sleep(wait_time)
            process_single_row(group_id, scene_id, video_path, num_frames, fps, prompt_text, model, api_key, domain_file)
            task_queue.task_done()
            pbar.update(1)

    # Fill task queue
    task_queue = manager.Queue()
    for idx, row in enumerate(rows):
        task_queue.put((idx, row))

    start_time = time.time() if record_time else None
    with tqdm(total=len(rows), desc=f"({group_id})VQA Finished") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                futures.append(executor.submit(worker, task_queue, pbar))
            for future in concurrent.futures.as_completed(futures):
                future.result()
    if record_time:
        total_time = time.time() - start_time
    if record_time and len(rows) > 0:
        print(f"Total API time: {total_time:.2f}s, average per scene: {total_time/len(rows):.2f}s")
    elif record_time:
        print("No valid scenes, cannot calculate time.")

def parse_args():
    """
    Parse command line arguments for VQA batch processing.
    """
    parser = argparse.ArgumentParser(description='VQA batch processing script')
    parser.add_argument('--group_id', type=str, help='Group id')
    parser.add_argument('--prompt_file', type=str,
                        default=os.path.join(os.path.dirname(__file__), "vqa_prompt.txt"),
                        help='Prompt file path')
    parser.add_argument('--model', type=str, default="gemini-2.0-flash",
                        help='Model name')
    parser.add_argument('--api_key', type=str,
                        default="sk-****",
                        help='API key')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads')
    parser.add_argument('--wait_time', type=float, default=0.8,
                        help='Request interval for each thread (seconds)')
    parser.add_argument('--domain_file', type=str, default="/path/to/vqa_base_domain.txt",
                        help='API domain file path')
    parser.add_argument('--record_time', action='store_true',
                        help='Record processing time')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process(
        group_id=args.group_id,
        prompt_file=args.prompt_file,
        model=args.model,
        api_key=args.api_key,
        domain_file=args.domain_file,
        num_workers=args.num_workers,
        wait_time=args.wait_time,
        record_time=args.record_time
    )