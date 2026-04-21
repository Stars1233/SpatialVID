"""
Precise frame-level video cutting tool
Strategy: Two-phase seek + forced keyframe alignment output
"""

import argparse
import os
import concurrent.futures
from functools import partial
import pandas as pd
import subprocess
from scenedetect import FrameTimecode
from tqdm import tqdm


FFMPEG_PATH = "/usr/local/bin/ffmpeg"


def get_ffmpeg_acceleration():
    try:
        output = subprocess.check_output(
            [FFMPEG_PATH, "-encoders"], stderr=subprocess.DEVNULL
        ).decode("utf-8")
        if "hevc_nvenc" in output:
            return "nvidia"
        return "cpu"
    except Exception as e:
        print(f"FFmpeg acceleration detection failed: {e}")
        return "cpu"


ACCELERATION_TYPE = get_ffmpeg_acceleration()
print(f"FFmpeg acceleration type: {ACCELERATION_TYPE}")


# ════════════════════════════════════════════════════════════
# Core Utility Functions
# ════════════════════════════════════════════════════════════

def seconds_to_timecode(seconds: float) -> str:
    """
    Convert seconds to FFmpeg precise timecode string.
    Keep enough decimal places to ensure frame accuracy.

    Example: 1.033333 -> "0:00:01.033333"
    """
    hours   = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs    = seconds % 60
    # Keep 6 decimal places (microsecond-level precision)
    return f"{hours}:{minutes:02d}:{secs:09.6f}"


def build_precise_cut_cmd(
    video_path:   str,
    start_sec:    float,
    end_sec:      float,
    save_path:    str,
    args,
    process_id:   int,
    shorter_size: int | None,
) -> list[str]:
    """
    Build frame-precise FFmpeg cut command.

    Strategy: Two-phase seek
    ┌──────────────────────────────────────────────────────────┐
    │ -ss (pre, coarse seek)                                 │
    │   -> Jump to nearest keyframe before start_sec         │
    │   -> Avoid decoding from file start (speed optimize)   │
    │                                                          │
    │ -i input                                                 │
    │                                                          │
    │ -ss (post, fine seek)                                   │
    │   -> Decode from coarse point to exact start_sec       │
    │   -> value = start_sec - coarse_seek (always positive) │
    │                                                          │
    │ -t duration                                              │
    │   -> Exact duration                                     │
    │                                                          │
    │ Force re-encode (cannot use -c copy, otherwise         │
    │ start frame won't be precise)                          │
    └──────────────────────────────────────────────────────────┘
    """
    duration = end_sec - start_sec

    if duration <= 0:
        raise ValueError(f"Invalid duration {duration:.4f}s (start={start_sec}, end={end_sec})")

    # ==== Phase 1: Coarse seek (pre seek) ====
    # Safety margin: ensure coarse point is before start_sec keyframe
    # Too little -> may land after start_sec (seek ineffective)
    # Too much -> decode more frames (slightly slower)
    # Experience: max(GOP_size, 5s) covers most videos
    GOP_SAFETY_MARGIN = 5.0
    coarse_seek = max(0.0, start_sec - GOP_SAFETY_MARGIN)

    # Offset for post precise seek = target time - coarse time
    fine_seek = start_sec - coarse_seek

    cmd = [FFMPEG_PATH, "-nostdin", "-y"]

    # ==== GPU hardware acceleration (decode phase) ====
    if ACCELERATION_TYPE == "nvidia":
        cmd += [
            "-hwaccel",               "cuda",
            "-hwaccel_output_format", "cuda",
            "-hwaccel_device",        str(process_id % args.gpu_num),
        ]

    # ==== Phase 1: Coarse seek (pre, fast jump to GOP boundary) ====
    cmd += ["-ss", seconds_to_timecode(coarse_seek)]

    # ==== Input file ====
    cmd += ["-i", video_path]

    # ==== Phase 2: Precise seek (post, decode from GOP boundary to exact frame) ====
    # Only need post seek when fine_seek > 0
    # When coarse_seek == 0, fine_seek == start_sec, still correct
    if fine_seek > 0.001:  # Ignore errors less than 1ms
        cmd += ["-ss", seconds_to_timecode(fine_seek)]

    # ==== Exact duration ====
    cmd += ["-t", seconds_to_timecode(duration)]

    # ==== Video filters (scale + fps) ====
    filters = _build_video_filters(shorter_size, args, ACCELERATION_TYPE)
    if filters:
        cmd += ["-vf", ",".join(filters)]

    # ==== Encoder (must re-encode to ensure frame precision) ====
    cmd += _build_encoder_args(ACCELERATION_TYPE)

    # ==== Frame rate ====
    if args.target_fps is not None:
        cmd += ["-r", str(args.target_fps)]

    # ==== Audio ====
    if args.keep_audio:
        cmd += ["-map", "0:v", "-map", "0:a?", "-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-map", "0:v", "-an"]

    # ==== Output: force keyframe at first frame for easy concatenation/playback ====
    cmd += [
        "-force_key_frames", "expr:gte(t,0)",  # Force keyframe at second 0
        save_path,
    ]

    return cmd


def _build_video_filters(shorter_size, args, accel_type) -> list[str]:
    """Build video filter list"""
    filters = []

    if shorter_size is not None:
        if accel_type == "nvidia":
            # CUDA scale filter
            scale = (
                f"scale_cuda="
                f"'if(gt(iw,ih),-2,{shorter_size})':"
                f"'if(gt(iw,ih),{shorter_size},-2)'"
            )
        else:
            # Software scale: lanczos best quality, bicubic next
            scale = (
                f"scale="
                f"'if(gt(iw,ih),-2,{shorter_size})':"
                f"'if(gt(iw,ih),{shorter_size},-2)'"
                f":flags=lanczos"
            )
        filters.append(scale)

    if args.target_fps is not None:
        # fps filter more accurate than -r parameter (-r sometimes drops frames)
        filters.append(f"fps={args.target_fps}")

    return filters


def _build_encoder_args(accel_type) -> list[str]:
    """Build encoder arguments"""
    if accel_type == "nvidia":
        return [
            "-c:v", "hevc_nvenc",
            "-preset",  "p4",      # p4=quality/speed balance, p7=slowest best
            "-rc",      "vbr",
            "-cq",      "24",      # Quality factor, smaller is better (like CRF)
            "-b:v",     "0",       # No bitrate limit in VBR mode
        ]
    else:
        return [
            "-c:v", "libx264",
            "-preset", "fast",     # fast is best speed/quality for precise cutting
            "-crf",    "18",       # High quality (0=lossless, 23=default, 18=visually lossless)
            "-pix_fmt", "yuv420p", # Most compatible pixel format
        ]


# ════════════════════════════════════════════════════════════
# Single Row Processing (maintains compatibility with original interface)
# ════════════════════════════════════════════════════════════

def process_single_row(row, args, process_id):
    """
    Precise frame-level cutting of a single segment.

    Returns:
        (row_values_list, valid, error_message)
    """
    video_path = row["video_path"]
    save_dir   = args.video_save_dir

    #
    # ==== Scale size calculation ====
    shorter_size = args.shorter_size
    if (shorter_size is not None) and ("height" in row) and ("width" in row):
        min_size = min(row["height"], row["width"])
        if min_size <= shorter_size:
            shorter_size = None  # Already small enough, skip scaling (no upsample)

    # ==== Timestamp parsing ====
    try:
        seg_start = FrameTimecode(timecode=row["timestamp_start"], fps=row["fps"])
        seg_end   = FrameTimecode(timecode=row["timestamp_end"],   fps=row["fps"])
    except Exception as e:
        error_msg = f"Invalid timestamp for id={row.get('id', '?')}: {e}"
        print(error_msg)
        return row.values.tolist(), False, error_msg

    start_sec = seg_start.get_seconds()
    end_sec   = seg_end.get_seconds()
    duration  = end_sec - start_sec

    if duration <= 0:
        error_msg = (
            f"Invalid duration {duration:.4f}s for id={row.get('id','?')} "
            f"(start={start_sec:.4f}, end={end_sec:.4f})"
        )
        print(error_msg)
        return row.values.tolist(), False, error_msg

    clip_id   = row["id"]
    save_path = os.path.join(save_dir, f"{clip_id}.mp4")

    # ==== Skip if already exists ====
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        row = row.copy()
        row["video_path"] = save_path
        return row.values.tolist(), True, ""

    # ==== Source file check ====
    if not os.path.exists(video_path):
        error_msg = f"Source video not found: {video_path} (id={clip_id})"
        print(error_msg)
        return row.values.tolist(), False, error_msg

    # ==== Build precise cut command ====
    try:
        cmd = build_precise_cut_cmd(
            video_path   = video_path,
            start_sec    = start_sec,
            end_sec      = end_sec,
            save_path    = save_path,
            args         = args,
            process_id   = process_id,
            shorter_size = shorter_size,
        )
    except ValueError as e:
        error_msg = f"Command build failed for id={clip_id}: {e}"
        print(error_msg)
        return row.values.tolist(), False, error_msg

    # ==== Execute FFmpeg ====
    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
        error_msg   = f"FFmpeg failed for id={clip_id}:\n{stderr_text}"
        print(error_msg)
        _cleanup(save_path)
        return row.values.tolist(), False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error for id={clip_id}: {e}"
        print(error_msg)
        _cleanup(save_path)
        return row.values.tolist(), False, error_msg

    # ==== Basic integrity check ====
    if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
        _cleanup(save_path)
        error_msg = f"FFmpeg produced empty/missing output for id={clip_id}"
        print(error_msg)
        return row.values.tolist(), False, error_msg

    row = row.copy()
    row["video_path"] = save_path
    return row.values.tolist(), True, ""


def _cleanup(path: str):
    """Safely delete file"""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# ════════════════════════════════════════════════════════════
# Argument Parsing
# ════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Precise frame-level video cutting tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ==== Input/Output ====
    parser.add_argument("--csv_path",      type=str, required=True,
                        help="Input CSV file path")
    parser.add_argument("--csv_save_path", type=str, required=True,
                        help="Output CSV file path (success records)")
    parser.add_argument("--video_save_dir", type=str, required=True,
                        help="Directory to save cut segments")

    # ==== Video parameters ====
    parser.add_argument("--target_fps",   type=int,   default=None,
                        help="Target frame rate (None=keep source frame rate)")
    parser.add_argument("--shorter_size", type=int,   default=None,
                        help="Short edge target size (maintain aspect ratio, no upsample)")
    parser.add_argument("--keep_audio",   action="store_true",
                        help="Keep audio track (default: discard)")

    # ==== Parallel control ====
    parser.add_argument("--num_workers",      type=int, default=None,
                        help="Number of parallel workers (None=auto=CPU cores)")
    parser.add_argument("--disable_parallel", action="store_true",
                        help="Disable parallel processing (for debugging)")
    parser.add_argument("--gpu_num",          type=int, default=1,
                        help="Number of available GPUs")

    # ==== Result handling ====
    parser.add_argument("--drop_invalid_timestamps", action="store_true",
                        help="Filter invalid timestamps and save corrected CSV")

    return parser.parse_args()


# ════════════════════════════════════════════════════════════
# Parallel Worker
# ════════════════════════════════════════════════════════════

def _worker_fn(task: tuple, args, process_id: int) -> tuple:
    """
    Top-level worker function for ProcessPoolExecutor (must be serializable).

    Args:
        task: (index, row_dict)  <- Use dict instead of Series to avoid serialization issues

    Returns:
        (index, row_values, valid, error_msg)
    """
    index, row_dict = task

    # Restore dict to pandas Series (process_single_row depends on Series interface)
    row = pd.Series(row_dict)
    return (index,) + tuple(process_single_row(row, args, process_id)[0:3])
    # Note: process_single_row returns (row_values, valid, error_msg)
    # Packed here as (index, row_values, valid, error_msg)


# ════════════════════════════════════════════════════════════
# Result Saving
# ════════════════════════════════════════════════════════════

def save_results(all_results: list, csv: pd.DataFrame, args):
    """
    Save processing results to success/failure CSVs separately.

    Success CSV: Remove timestamp helper columns, update video_path to cut path
    Failure CSV: Keep all original columns, add error column
    """
    columns = csv.columns.tolist()

    success_rows, failed_rows, failed_errors = [], [], []

    for index, row_values, valid, error_msg in all_results:
        if valid:
            success_rows.append(row_values)
        else:
            failed_rows.append(row_values)
            failed_errors.append(error_msg)

    # ==== Save success records ====
    if success_rows:
        success_df = pd.DataFrame(success_rows, columns=columns)

        # Remove cutting process helper columns (not needed by downstream)
        drop_cols = [
            c for c in ["timestamp_start", "timestamp_end", "frame_start", "frame_end"]
            if c in success_df.columns
        ]
        if drop_cols:
            success_df = success_df.drop(columns=drop_cols)

        success_df.to_csv(args.csv_save_path, index=False)
        print(f"\n[OK] Success: {len(success_df)} records -> {args.csv_save_path}")
    else:
        print("\n[X] No success records")

    # ==== Save failure records ====
    if failed_rows:
        base, ext       = os.path.splitext(args.csv_save_path)
        failed_csv_path = f"{base}_failed{ext}"

        failed_df          = pd.DataFrame(failed_rows, columns=columns)
        failed_df["error"] = failed_errors
        failed_df.to_csv(failed_csv_path, index=False)
        print(f"[X] Failed: {len(failed_df)} records -> {failed_csv_path}")

    # ==== Save corrected timestamps (optional) ====
    if args.drop_invalid_timestamps and failed_rows:
        valid_indices   = [r[0] for r in all_results if r[2]]
        filtered_csv    = csv.iloc[valid_indices]
        assert args.csv_path.endswith("timestamp.csv"), \
            "--drop_invalid_timestamps only supports *timestamp.csv files"
        corrected_path  = args.csv_path.replace("timestamp.csv", "correct_timestamp.csv")
        filtered_csv.to_csv(corrected_path, index=False)
        print(f"[OK] Corrected timestamps -> {corrected_path}")


# ════════════════════════════════════════════════════════════
# Main Function
# ════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # ==== Pre-check ====
    if not os.path.exists(args.csv_path):
        print(f"[ERROR] CSV file does not exist: {args.csv_path}")
        return

    os.makedirs(args.video_save_dir, exist_ok=True)

    csv = pd.read_csv(args.csv_path)
    total = len(csv)
    print(f"Total {total} records to process")

    all_results = []

    # ==== Serial mode ====
    if args.disable_parallel:
        for index, row in tqdm(csv.iterrows(), total=total, desc="Cutting progress"):
            row_values, valid, error_msg = process_single_row(row, args, process_id=0)
            all_results.append((index, row_values, valid, error_msg))

    # ==== Parallel mode ====
    else:
        num_workers = args.num_workers or (os.cpu_count() or 1)
        num_workers = min(num_workers, total)  # worker count not exceeding task count

        # Convert row to dict to avoid pandas Series serialization issues
        tasks = [
            (index, row.to_dict())
            for index, row in csv.iterrows()
        ]

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            # Use enumerate to round-robin process_id (GPU rotation)
            futures = {
                executor.submit(
                    _worker_fn,
                    task,
                    args,
                    task_idx % max(args.gpu_num, 1),  # GPU rotation
                ): task_idx
                for task_idx, task in enumerate(tasks)
            }

            with tqdm(total=total, desc="Cutting progress") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()      # (index, row_values, valid, error_msg)
                        all_results.append(result)
                    except Exception as e:
                        task_idx    = futures[future]
                        index, _    = tasks[task_idx]
                        row_values  = csv.iloc[index].values.tolist()
                        all_results.append((index, row_values, False, str(e)))
                        print(f"\n[ERROR] Worker exception (task_idx={task_idx}): {e}")
                    finally:
                        pbar.update(1)

    # ==== Sort by original order ====
    all_results.sort(key=lambda x: x[0])

    # ==== Statistics summary ====
    success_count = sum(1 for r in all_results if r[2])
    failed_count  = total - success_count
    print(f"\n{'='*50}")
    print(f"Processing complete: Total={total}, Success={success_count}, Failed={failed_count}")
    print(f"{'='*50}")

    # ==== Save results ====
    save_results(all_results, csv, args)


if __name__ == "__main__":
    main()
