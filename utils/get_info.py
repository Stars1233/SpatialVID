"""
Video information extraction utility supporting multiple backends (OpenCV, TorchVision, AV).
"""

import argparse
import os
import random
import queue
from glob import glob
import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures


def get_video_length(cap, method="header"):
    """Get video frame count using different methods"""
    assert method in ["header", "set"]
    if method == "header":
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        length = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    return length


def get_video_info(args):
    """Extract video information using specified backend"""
    idx, path, backend = args
    try:
        if backend == "torchvision":
            from tools.utils.read_video import read_video

            vframes, infos = read_video(path)
            num_frames, height, width = (
                vframes.shape[0],
                vframes.shape[2],
                vframes.shape[3],
            )
        elif backend == "opencv":
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError("Video open failed")
            num_frames = get_video_length(cap, method="header")
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
        elif backend == "av":
            import av

            container = av.open(path)
            stream = container.streams.video[0]
            num_frames = int(stream.frames)
            height = int(stream.height)
            width = int(stream.width)
        else:
            raise ValueError("Unknown backend")

        # Calculate derived metrics
        hw = height * width
        aspect_ratio = height / width if width > 0 else np.nan
        return (idx, True, num_frames, height, width, aspect_ratio, hw)
    except Exception as e:
        return (idx, False, 0, 0, 0, np.nan, np.nan)


def read_file(input_path):
    """Read CSV or Parquet file"""
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")


def save_file(data, output_path):
    """Save DataFrame to CSV or Parquet file"""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir) and output_dir != "":
        os.makedirs(output_dir)
    if output_path.endswith(".csv"):
        return data.to_csv(output_path, index=False)
    elif output_path.endswith(".parquet"):
        return data.to_parquet(output_path, index=False)
    else:
        raise NotImplementedError(f"Unsupported file format: {output_path}")


def read_data(input_paths):
    """Load and concatenate multiple data files"""
    data = []
    input_name = ""
    input_list = []
    for input_path in input_paths:
        input_list.extend(glob(input_path))
    print("Input files:", input_list)

    for i, input_path in enumerate(input_list):
        if not os.path.exists(input_path):
            continue
        data.append(read_file(input_path))
        input_name += os.path.basename(input_path).split(".")[0]
        if i != len(input_list) - 1:
            input_name += "+"
        print(f"Loaded {len(data[-1])} samples from '{input_path}'.")

    if len(data) == 0:
        print(f"No samples to process. Exit.")
        exit()

    data = pd.concat(data, ignore_index=True, sort=False)
    print(f"Total number of samples: {len(data)}")
    return data, input_name


def worker(task_queue, results_queue, args):
    """Worker function for parallel video processing"""
    while True:
        try:
            index, path, backend = task_queue.get(timeout=1)
        except queue.Empty:
            break
        result = get_video_info((index, path, backend))
        results_queue.put((index, result))
        task_queue.task_done()


def main(args):
    """Main function to extract video information"""
    # Load data
    data, input_name = read_data(args.input)

    # Get output path
    output_path = get_output_path(args, input_name)

    # Setup multiprocessing
    from multiprocessing import Manager

    manager = Manager()
    task_queue = manager.Queue()
    results_queue = manager.Queue()

    for index, row in data.iterrows():
        task_queue.put((index, row["video_path"], args.backend))

    num_workers = args.num_workers if args.num_workers else os.cpu_count()

    # Process videos in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for _ in range(num_workers):
            future = executor.submit(worker, task_queue, results_queue, args)
            futures.append(future)

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Workers completing",
        ):
            future.result()

    # Collect and process results
    ret = []
    while not results_queue.empty():
        ret.append(results_queue.get())

    ret.sort(key=lambda x: x[0])
    ret = [x[1] for x in ret]
    (
        idx_list,
        success_list,
        num_frames_list,
        height_list,
        width_list,
        aspect_ratio_list,
        hw_list,
    ) = zip(*ret)

    # Add extracted information to DataFrame
    data["success"] = success_list
    data["num_frames"] = num_frames_list
    data["height"] = height_list
    data["width"] = width_list
    data["aspect_ratio"] = aspect_ratio_list
    data["resolution"] = hw_list

    # Filter existing files if requested
    if args.ext:
        assert "video_path" in data.columns
        data = data[data["video_path"].apply(os.path.exists)]

    # Sort by frame count
    if "num_frames" in data.columns:
        data = data.sort_values(by="num_frames", ascending=True)

    save_file(data, output_path)
    print(f"Saved {len(data)} samples to {output_path}.")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract video information using multiple backends"
    )
    parser.add_argument(
        "input", type=str, nargs="+", help="Path to input dataset files"
    )
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument(
        "--backend",
        type=str,
        default="opencv",
        help="Video backend",
        choices=["opencv", "torchvision", "av"],
    )
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        help="Output format",
        choices=["csv", "parquet"],
    )
    parser.add_argument(
        "--disable-parallel", action="store_true", help="Disable parallel processing"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None, help="Number of parallel workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # File existence checking
    parser.add_argument("--ext", action="store_true", help="Check if video files exist")

    return parser.parse_args()


def get_output_path(args, input_name):
    """Generate output file path based on input and arguments"""
    if args.output is not None:
        return args.output

    name = input_name
    dir_path = os.path.dirname(args.input[0])
    print("dir path", dir_path)

    # Add suffix based on options
    name += "_info"
    if args.ext:
        name += "_ext"

    output_path = os.path.join(dir_path, f"{name}.{args.format}")
    return output_path


if __name__ == "__main__":
    args = parse_args()
    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    main(args)
