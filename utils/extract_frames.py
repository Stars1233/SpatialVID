import os
import sys
import cv2
import argparse
import pandas as pd
import queue
import concurrent.futures
from multiprocessing import Manager
from tqdm import tqdm

def extract_frames(video_path, output_folder, interval, frame_start, num_frames, target_size=None):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    for frame in range(num_frames):
        ret, image = cap.read()
        if not ret:
            break
        # Save the frame as an image file
        if frame % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame:06d}.jpg")
            if target_size is not None:
                image = cv2.resize(image, target_size)
            cv2.imwrite(frame_filename, image)

    cap.release()

def process_single_row(row, row_index, args):
    video_path = row["video_path"]
    if "frame_start" not in row:
        frame_start = 0
    else:
        frame_start = row["frame_start"]
    num_frames = row["num_frames"]
    output_folder = os.path.join(args.output_folder, row["id"])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_folder = os.path.join(output_folder, "img")

    if args.interval is None:
        interval = row["num_frames"] // 3
    else:
        interval = int(args.interval * row["fps"])

    extract_frames(video_path, output_folder, interval, frame_start, num_frames, args.target_size)

def worker(task_queue, args):
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        process_single_row(row,index,args)
        task_queue.task_done()

def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("csv_path", type=str, help="Path to the csv file")
    parser.add_argument("-o", "--output_folder", type=str, default="extract_frames", help="Output folder for extracted frames (default: extract_frames)")
    parser.add_argument("-i", "--interval", type=float, default=None, help="Interval for frame extraction")
    parser.add_argument("--target_size", type=str, default=None, help="Resize the frame to this size (width, height)")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of workers for parallel processing")
    parser.add_argument("--disable_parallel", action="store_true", help="Disable parallel processing")

    return parser.parse_args()

def main():
    args = parse_args()

    if args.target_size is not None:
        args.target_size = tuple(map(int, args.target_size.split("*")))

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    meta = pd.read_csv(args.csv_path)

    if args.disable_parallel:
        for index, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing rows"):
            process_single_row(row, index, args)
    else:
        if args.num_workers is not None:
            num_workers = args.num_workers
        else:
            num_workers = os.cpu_count()

        manager = Manager()
        task_queue = manager.Queue()

        for index, row in meta.iterrows():
            task_queue.put((index, row))

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _ in range(num_workers):
                future = executor.submit(worker, task_queue, args)
                futures.append(future)
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Finished workers"):
                future.result() 

if __name__ == "__main__":
    main()