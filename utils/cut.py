import argparse
import os
import queue
import concurrent.futures
from functools import partial
import pandas as pd
import subprocess
from scenedetect import FrameTimecode
from tqdm import tqdm

def get_ffmpeg_acceleration():
    """
    auto detect the best acceleration method
    NVIDIA > CPU。
    """
    try:
        # get the list of available encoders
        output = subprocess.check_output(["ffmpeg", "-encoders"], stderr=subprocess.DEVNULL).decode("utf-8")

        if "hevc_nvenc" in output:  # NVIDIA GPU
            return "nvidia"
        else:
            return "cpu"  # use CPU
    except Exception as e:
        print(f"FFmpeg acceleration detection failed: {e}")
        return "cpu"


ACCELERATION_TYPE = get_ffmpeg_acceleration()
print(f"FFmpeg acceleration type: {ACCELERATION_TYPE}")


def process_single_row(row, args, process_id):
    video_path = row["video_path"]

    save_dir = args.video_save_dir

    shorter_size = args.shorter_size
    if (shorter_size is not None) and ("height" in row) and ("width" in row):
        min_size = min(row["height"], row["width"])
        if min_size <= shorter_size:
            shorter_size = None

    seg_start = FrameTimecode(timecode=row["timestamp_start"], fps=row["fps"])
    seg_end = FrameTimecode(timecode=row["timestamp_end"], fps=row["fps"])

    id = row["id"]

    save_path = os.path.join(save_dir, f"{id}.mp4")
    if os.path.exists(save_path):
        # 将原本的video_path替换为save_path
        row["video_path"] = save_path
    else:
        try:
            # 构建输入流
            cmd = [
                args.ffmpeg_path,
                "-nostdin",
                "-y"
            ]
            if ACCELERATION_TYPE == "nvidia":
                cmd += [
                    "-hwaccel", "cuda",
                    "-hwaccel_output_format", "cuda",
                    "-hwaccel_device", str(process_id % args.gpu_num),
                ]
                
            cmd += [
                "-i", video_path,
                "-ss", str(seg_start.get_seconds()),
                "-to", str(seg_end.get_seconds())
            ]

            if ACCELERATION_TYPE == "nvidia":
                cmd += [
                    "-c:v", "hevc_nvenc",
                    "-preset", "fast"
                ]
            else:
                cmd += [
                    "-c:v", "libx264",
                    "-preset", "fast"
                ]

            if args.target_fps is not None:
                cmd += [
                    "-r", str(args.target_fps)
                ]

            if shorter_size is not None:
                if ACCELERATION_TYPE == "nvidia":
                    cmd += [
                        "-vf", f"scale_cuda='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'"
                    ]
                else:
                    cmd += [
                        "-vf", f"scale='if(gt(iw,ih),-2,{shorter_size})':'if(gt(iw,ih),{shorter_size},-2)'"
                    ]

            cmd += [
                "-map", "0:v",
                save_path
                ]

            subprocess.run(cmd, check=True, stderr=subprocess.PIPE)

            row["video_path"] = save_path
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg command failed: {e.stderr.decode('utf-8')}")

    return row.values.tolist(), True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("--video_save_dir", type=str, required=True, help="Directory to save the video clips")
    parser.add_argument("--csv_save_dir", type=str, required=True, help="Directory to save the CSV file")
    parser.add_argument("--ffmpeg_path", type=str, default="ffmpeg", help="Path to the ffmpeg executable")
    parser.add_argument("--target_fps", type=int, default=None, help="target fps of clips")
    parser.add_argument("--shorter_size", type=int, default=None, help="resize the shorter size by keeping ratio; will not do upscale")
    parser.add_argument("--num_workers", type=int, default=None, help="#workers for concurrent.futures")
    parser.add_argument("--disable_parallel", action="store_true", help="disable parallel processing")
    parser.add_argument("--drop_invalid_timestamps", action="store_true", help="drop rows with invalid timestamps")
    parser.add_argument("--gpu_num", type=int, default=1, help="gpu number")
    return parser.parse_args()


def worker(task_queue, results_queue, args, process_id):
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        row, valid = process_single_row(row, args, process_id)
        results_queue.put((index, row, valid))
        task_queue.task_done()


def main():
    args = parse_args()
    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        return

    os.makedirs(args.video_save_dir, exist_ok=True)
    os.makedirs(args.csv_save_dir, exist_ok=True)

    meta = pd.read_csv(args.meta_path)

    from multiprocessing import Manager
    manager = Manager()
    task_queue = manager.Queue()
    results_queue = manager.Queue()

    for index, row in meta.iterrows():
        task_queue.put((index, row))

    process_single_row_partial = partial(process_single_row, args=args)

    if args.disable_parallel:
        results = []
        for index, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing rows"):
            result = process_single_row_partial(row, index)
            results.append(result)
    else:
        if args.num_workers is not None:
            num_workers = args.num_workers
        else:
            num_workers = os.cpu_count()

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for id in range(num_workers):
                futures.append(executor.submit(worker, task_queue, results_queue, args, id))

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Finished workers"):
                future.result()

    results = []
    while not results_queue.empty():
        results.append(results_queue.get())

    results.sort(key=lambda x: x[0])
    results = [x[1:] for x in results]

    new_rows = []
    valid_rows = []
    for new_row_list, valid in results:
        new_rows.append(new_row_list)
        valid_rows.append(valid)

    if args.drop_invalid_timestamps:
        meta = meta[valid_rows]
        assert args.meta_path.endswith("timestamp.csv"), "Only support *timestamp.csv"
        meta.to_csv(args.meta_path.replace("timestamp.csv", "correct_timestamp.csv"), index=False)
        print(f"Corrected timestamp file saved to '{args.meta_path.replace('timestamp.csv', 'correct_timestamp.csv')}'")

    # 创建新的 DataFrame 并保存为 CSV 文件
    columns = meta.columns  # 获取原始 DataFrame 的列名
    new_df = pd.DataFrame(new_rows, columns=columns)
    new_csv_path = os.path.join(args.csv_save_dir, "clips_info_filtered.csv")
    new_df.to_csv(new_csv_path, index=False)
    print(f"Saved {len(new_df)} clip information to {new_csv_path}.")


if __name__ == "__main__":
    main()