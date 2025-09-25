"""
Batch inference script for CVD (Camera View Depth) optimization.
Processes multiple video clips in parallel using multi-GPU setup.
"""

import pandas as pd
import os
import argparse
import concurrent.futures
from multiprocessing import Manager
import subprocess
import queue
from tqdm import tqdm


def process_single_row(row, index, args, worker_id=0):
    """Process a single video clip for CVD optimization."""
    dir_path = os.path.join(args.dir_path, row["id"])
    device_id = worker_id % args.gpu_num

    # Build command for CVD optimization with specific GPU
    cmd = (
        f"CUDA_VISIBLE_DEVICES={args.gpu_id[device_id]} python camera_pose_annotation/cvd_opt/cvd_opt.py "
        f"--dir_path {dir_path} "
        f"--w_grad 2.0 --w_normal 5.0 "
    )
    if args.only_depth:
        cmd += "--only_depth "
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error optimizing CVD for {row['id']}: {stderr.decode()}")


def worker(task_queue, args, worker_id, pbar):
    """Worker function for parallel CVD optimization processing."""
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        process_single_row(row, index, args, worker_id)
        task_queue.task_done()
        pbar.update(1)


def parse_args():
    """Parse command line arguments for CVD batch processing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, help="Path to the csv file")
    parser.add_argument("--dir_path", type=str, default="./outputs")
    parser.add_argument("--only_depth", action="store_true", help="Only save optimized depth")
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="Comma-separated list of GPU IDs to use"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for parallel processing",
    )
    parser.add_argument(
        "--disable_parallel", action="store_true", help="Disable parallel processing"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse GPU configuration
    args.gpu_num = len(args.gpu_id.split(","))
    args.gpu_id = [int(gpu) for gpu in args.gpu_id.split(",")]

    df = pd.read_csv(args.csv_path)

    if args.disable_parallel:
        # Sequential processing
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            process_single_row(row, index, args)
    else:
        # Parallel processing with multiple workers
        manager = Manager()
        task_queue = manager.Queue()
        for index, row in df.iterrows():
            task_queue.put((index, row))
        with tqdm(total=len(df), desc="Processing rows") as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.num_workers
            ) as executor:
                futures = []
                for id in range(args.num_workers):
                    futures.append(executor.submit(worker, task_queue, args, id, pbar))

                for future in concurrent.futures.as_completed(futures):
                    future.result()


if __name__ == "__main__":
    main()
