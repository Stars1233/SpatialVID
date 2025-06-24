import pandas as pd
import os
import argparse
import concurrent.futures
from multiprocessing import Manager
import subprocess
import queue
from tqdm import tqdm

def process_single_row(row, index, args, worker_id=0):
    dir_path = os.path.join(args.dir_path, row["id"])
    device_id = worker_id % args.gpu_num
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # =======================================================
    # ------------------- Extract Videos --------------------
    # =======================================================
    # Extract videos into frames
    if args.extract_videos or args.all:
        video_path = row["video_path"]
        if "frame_start" in row:
            frame_start = row["frame_start"]
        else:
            frame_start = 0

        if "num_frames" in row:
            num_frames = row["num_frames"]
        else:
            num_frames = 0

        interval = int(row["fps"] * args.extract_interval)
        
        cmd = (
            f"python tools/extract_video.py"
            f" --video_path {video_path} "
            f" --frame_start {frame_start} "
            f" --num_frames {num_frames} "
            f" --output_path {dir_path} "
            f" --interval {interval} "
        )
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error extracting frames for {row['id']}: {stderr.decode()}")

    # # =======================================================
    # # ------------------- Depth Generation ------------------
    # # =======================================================
    # # ----------------- Run Depth-Anything -----------------
    if args.depth_anything or args.all:
        cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu_id[device_id]} python Depth-Anything/run_videos.py "
            f"--input-size 518 "
            f"--load-from {args.checkpoints_path}/Depth-Anything/depth_anything_v2_{args.dpt_encoder}.pth "
            f"--dir_path {dir_path} "
            f"--encoder {args.dpt_encoder} "
        )
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error generating depth(Depth Anything) for {row['id']}: {stderr.decode()}")

    # # --------------------- Run UniDepth -------------------
    if args.unidepth or args.all:
        cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu_id[device_id]} python UniDepth/demo_mega-sam.py "
            f"--dir_path {dir_path} "
            f"--load-from {args.checkpoints_path}/UniDepth "
        )
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error generating depth(UniDepth) for {row['id']}: {stderr.decode()}")

    # =======================================================
    # ----------------- Run Camera Tracking -----------------
    # =======================================================
    if args.camera_tracking or args.all:
        cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu_id[device_id]} python camera_tracking_scripts/test_demo.py "
            f"--dir_path {dir_path} "
            f"--weights {args.checkpoints_path}/megasam_final.pth "
            f"--disable_vis"
        )
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error tracking camera for {row['id']}: {stderr.decode()}")
    # # =======================================================
    # # ------ Run Consistent Video Depth Optimization --------
    # # =======================================================
    # # ------------------- Run Optical Flow ------------------
    if args.optical_flow or args.all:
        cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu_id[device_id]} python cvd_opt/preprocess_flow.py "
            f"--dir_path {dir_path} "
            f"--model {args.checkpoints_path}/raft/raft-things.pth "
            f"--mixed_precision"
        )
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error generating optical flow for {row['id']}: {stderr.decode()}")
    
    # # ---------------- Run CVD optmization ------------------
    if args.cvd_opt or args.all:
        cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu_id[device_id]} python cvd_opt/cvd_opt.py "
            f"--dir_path {dir_path} "
            f"--w_grad 2.0 --w_normal 5.0 "
        )
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Error optimizing CVD for {row['id']}: {stderr.decode()}")


def worker(task_queue, args, worker_id):
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        process_single_row(row, index, args, worker_id)
        task_queue.task_done()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str, help='Path to the csv file')
    parser.add_argument('--dir_path', type=str, default='./outputs')
    parser.add_argument('--checkpoints_path', type=str, default='./checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='Comma-separated list of GPU IDs to use')

    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for parallel processing')
    parser.add_argument('--disable_parallel', action='store_true', help='Disable parallel processing')

    parser.add_argument('--extract_interval', type=float, default=None, help='Interval seconds for extracting frames from video')
    parser.add_argument('--dpt_encoder', type=str, default=None, choices=['vits', 'vitb', 'vitl', 'vitg'], help='Encoder for Depth Anything')
    
    parser.add_argument('--extract_videos', action='store_true', help='Extract videos into frames')
    parser.add_argument('--depth_anything', action='store_true', help='Run Depth Anything')
    parser.add_argument('--unidepth', action='store_true', help='Run UniDepth')
    parser.add_argument('--camera_tracking', action='store_true', help='Run camera tracking')
    parser.add_argument('--cvd_opt', action='store_true', help='Run Consistent Video Depth Optimization')
    parser.add_argument('--optical_flow', action='store_true', help='Run Optical Flow')
    parser.add_argument('--all', action='store_true', help='Run all processes')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if (args.extract_videos or args.all) and args.extract_interval is None:
        parser.error("--extract_videos is set, but --extract_interval is not provided.")
        
    if (args.depth_anything or args.all) and args.dpt_encoder is None:
        parser.error("--depth_anything is set, but --dpt_encoder is not provided.")

    args.gpu_num = len(args.gpu_id.split(','))
    args.gpu_id = [int(gpu) for gpu in args.gpu_id.split(',')]

    df = pd.read_csv(args.csv_path)

    csv_name = os.path.basename(args.csv_path).split('.')[0]

    args.dir_path = os.path.join(args.dir_path, csv_name)
    if not os.path.exists(args.dir_path):
        os.makedirs(args.dir_path)

    if args.disable_parallel:
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            process_single_row(row, index, args)
    else:
        manager = Manager()
        task_queue = manager.Queue()
        for index, row in df.iterrows():
            task_queue.put((index, row))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for id in range(args.num_workers):
                futures.append(executor.submit(worker, task_queue, args, id))
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Finished workers"):
                future.result()


if __name__ == "__main__":
    main()
