import argparse
import os
import random
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures


# ======================================================
# read & write
# ======================================================
def read_file(input_path):
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")


def save_file(data, output_path):
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
    data = []
    input_list = []
    for input_path in input_paths:
        input_list.extend(glob(input_path))
    print("Input files:", input_list)
    for i, input_path in enumerate(input_list):
        if not os.path.exists(input_path):
            continue
        data.append(read_file(input_path))
        print(f"Loaded {len(data[-1])} samples from '{input_path}'.")
    if len(data) == 0:
        print(f"No samples to process. Exit.")
        exit()
    data = pd.concat(data, ignore_index=True, sort=False)
    print(f"Total number of samples: {len(data)}")
    return data


# ======================================================
# main
# ======================================================
def main(args):
    # reading data
    data = read_data(args.input)

    # filtering
    if args.frames_min is not None:
        assert "num_frames" in data.columns
        data = data[data["num_frames"] >= args.frames_min]
    if args.frames_max is not None:
        assert "num_frames" in data.columns
        data = data[data["num_frames"] <= args.frames_max]
    if args.fps_max is not None:
        assert "fps" in data.columns
        data = data[(data["fps"] <= args.fps_max) | np.isnan(data["fps"])]
    if args.fps_min is not None:
        assert "fps" in data.columns
        data = data[(data["fps"] >= args.fps_min) | np.isnan(data["fps"])]
    if args.resolution_max is not None:
        if "resolution" not in data.columns:
            height = data["height"]
            width = data["width"]
            data["resolution"] = height * width
        data = data[data["resolution"] <= args.resolution_max]
    if args.aes_min is not None:
        assert "aes" in data.columns
        data = data[data["aes"] >= args.aes_min]
    if args.ocr_max is not None:
        assert "ocr" in data.columns
        data = data[data["ocr"] <= args.ocr_max]
    if args.ocr_min is not None:
        assert "ocr" in data.columns
        data = data[data["ocr"] >= args.ocr_min]
    if args.lum_min is not None:
        assert "lum_mean" in data.columns
        data = data[data["lum_mean"] >= args.lum_min]
    if args.lum_max is not None:
        assert "lum_mean" in data.columns
        data = data[data["lum_mean"] <= args.lum_max]
    if args.blur_max is not None:
        assert "blur" in data.columns
        data = data[data["blur"] <= args.blur_max]
    if args.flow_min is not None:
        assert "flow_mean" in data.columns
        data = data[data["flow_mean"] >= args.flow_min]
    if args.flow_max is not None:
        assert "flow_mean" in data.columns
        data = data[data["flow_mean"] <= args.flow_max]
    if args.motion_min is not None:
        assert "motion" in data.columns
        data = data[data["motion"] >= args.motion_min]
    if args.motion_max is not None:
        assert "motion" in data.columns
        data = data[data["motion"] <= args.motion_max]

    csv_save_dir = os.path.dirname(args.input[0])
    output_path = os.path.join(csv_save_dir, "filtered_clips.csv")
    save_file(data, output_path)
    print(f"Saved {len(data)} samples to {output_path}.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, nargs="+", help="path to the input dataset")
    parser.add_argument("--format", type=str, default="csv", help="output format", choices=["csv", "parquet"])
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # score filtering
    parser.add_argument("--frames_min", type=int, default=None, help="filter the dataset by minimum number of frames")
    parser.add_argument("--frames_max", type=int, default=None, help="filter the dataset by maximum number of frames")
    parser.add_argument("--resolution_max", type=int, default=None, help="filter the dataset by maximum resolution")
    parser.add_argument("--aes_min", type=float, default=None, help="filter the dataset by minimum aes score")
    parser.add_argument("--flow_min", type=float, default=None, help="filter the dataset by minimum flow score")
    parser.add_argument("--flow_max", type=float, default=None, help="filter the dataset by maximum flow score")
    parser.add_argument("--ocr_max", type=float, default=None, help="filter the dataset by maximum ocr score")
    parser.add_argument("--ocr_min", type=float, default=None, help="filter the dataset by minimum ocr score")
    parser.add_argument("--fps_max", type=float, default=None, help="filter the dataset by maximum fps")
    parser.add_argument("--fps_min", type=float, default=None, help="filter the dataset by minimum fps")
    parser.add_argument("--lum_min", type=float, default=None, help="filter the dataset by minimum luminance score")
    parser.add_argument("--lum_max", type=float, default=None, help="filter the dataset by maximum luminance score")
    parser.add_argument("--blur_max", type=float, default=None, help="filter the dataset by maximum blur score")
    parser.add_argument("--motion_min", type=float, default=None, help="filter the dataset by minimum motion score")
    parser.add_argument("--motion_max", type=float, default=None, help="filter the dataset by maximum motion score")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    main(args)