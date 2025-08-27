import os
import argparse
import pandas as pd
import numpy as np
import torch
import concurrent.futures
import multiprocessing as mp
from multiprocessing import Manager
import queue
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix


def expand(loaded_data):
    """
    Reconstruct 3D mask from sparse matrix data
    """
    reconstructed_sparse_matrices = []
    num_frames = (len(loaded_data) - 1) // 3  # Calculate number of frames
    matrix_shape = loaded_data['shape']  # Get original matrix dimensions

    # Reconstruct sparse matrix for each frame
    for i in range(num_frames):
        data = loaded_data[f'f_{i}_data']
        indices = loaded_data[f'f_{i}_indices']
        indptr = loaded_data[f'f_{i}_indptr']
        reconstructed_matrix = csr_matrix((data, indices, indptr), shape=matrix_shape)
        reconstructed_sparse_matrices.append(reconstructed_matrix)

    # Stack all frames into a 3D array (frames, height, width)
    reconstructed_mask_3d = np.stack([m.toarray() for m in reconstructed_sparse_matrices], axis=0)
    return reconstructed_mask_3d


def load_file(cam_pos_file, mask_file, device):
    try:
        # Load camera parameters and split into position and rotation components
        params = torch.from_numpy(np.load(cam_pos_file)).float().to(device)
        cam_pos = params[:, :3]  # First 3 elements: position coordinates
        cam_rotate = params[:, 3:]  # Last 4 elements: rotation quaternions
        time_steps = params.shape[0]

        # Load and expand dynamic masks
        masks = torch.from_numpy(expand(np.load(mask_file))).to(device)
    except FileNotFoundError:
        print(f"Error: File not found - {cam_pos_file}")
        exit()
    except Exception as e:
        print(f"Error processing {cam_pos_file}: {e}")
        exit()

    return cam_pos, cam_rotate, time_steps, masks


def anomaly_detection(cam_pos, time_steps, threshold, device):
    if time_steps < 4:
        return True  # Not enough data to detect anomaly

    # Initialize prediction tensor
    preds = torch.zeros((time_steps, 3), dtype=torch.float32, device=device)
    error_count = 0

    # Linear prediction with acceleration
    for t in range(0, time_steps - 3):
        # Calculate velocity vectors
        v1 = cam_pos[t + 2] - cam_pos[t + 1]
        v2 = cam_pos[t + 1] - cam_pos[t]

        # Calculate acceleration
        acceleration = v1 - v2

        # Predict next position
        preds[t + 3] = cam_pos[t + 2] + v1 + 0.5 * acceleration

        # Calculate prediction error
        error = torch.sqrt(torch.sum((preds[t + 3] - cam_pos[t + 3]) ** 2))

        # Check against threshold
        if error > 0.03:
            error_count += 1
            if error_count >= threshold:
                return True
        else:
            error_count = 0  # Reset counter if error is within normal range

    return False


def move_distance(cam_pos, time_steps, device):
    total_distance = torch.tensor(0., dtype=torch.float32, device=device)

    # Distance thresholds for classification (adjust based on your data)
    high_threshold = 2.41
    medium_threshold = 0.92
    low_threshold = 0.28
    very_low_threshold = 0.08
    level = 0

    # Calculate cumulative distance between consecutive frames
    for i in range(0, time_steps - 1):
        total_distance += torch.norm(cam_pos[i + 1] - cam_pos[i])

    # Convert to Python float and determine level
    total_distance = total_distance.item()
    if total_distance >= high_threshold:
        level = 4
    elif total_distance >= medium_threshold:
        level = 3
    elif total_distance >= low_threshold:
        level = 2
    elif total_distance >= very_low_threshold:
        level = 1
    else:
        level = 0

    return total_distance, level


def quaternion_multiply(q1, q2):
    # Extract components (assuming q is in [x, y, z, w] format)
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]

    # Quaternion multiplication matrix
    matrix = torch.tensor([
        [w1, -z1, y1, x1],
        [z1, w1, -x1, y1],
        [-y1, x1, w1, z1],
        [-x1, -y1, -z1, w1]
    ], dtype=q1.dtype, device=q1.device)

    # Second quaternion as vector
    vector = torch.tensor([x2, y2, z2, w2], dtype=q2.dtype, device=q2.device)

    # Perform multiplication
    result = torch.matmul(matrix, vector)

    return result


def rotation_angle(cam_rotate, time_steps, device):
    total_radians = torch.tensor(0.0, device=device)

    # Calculate rotation between consecutive frames
    for i in range(0, time_steps - 1):
        q1 = cam_rotate[i]
        q2 = cam_rotate[i + 1]

        # Calculate inverse of first quaternion
        q1_inverse = torch.stack([-q1[0], -q1[1], -q1[2], q1[3]], dim=0)

        # Calculate relative rotation
        q_relative = quaternion_multiply(q2, q1_inverse)
        w = q_relative[3]

        # Clamp to handle numerical precision issues
        w = torch.clamp(w, -1.0, 1.0)

        # Convert to angle in radians
        rotation_angle_rad = 2 * torch.arccos(w)
        total_radians += rotation_angle_rad

    return total_radians.item()


def trajectory_turns(cam_pos, time_steps, device, threshold=0.45):
    if time_steps < 3:  # Need at least 3 points to detect turns
        return [], 0

    angles = []
    # Calculate angles between trajectory segments
    for t in range(1, time_steps - 1):
        # Vectors from current position to start and end points
        v1 = cam_pos[t] - cam_pos[0]
        v2 = cam_pos[time_steps - 1] - cam_pos[t]

        # Calculate vector norms (avoid division by zero)
        v1_norm = torch.norm(v1)
        v2_norm = torch.norm(v2)
        if v1_norm < 1e-8 or v2_norm < 1e-8:
            continue  # Skip zero-length vectors

        # Calculate cosine of the angle between vectors
        cos_theta = torch.dot(v1, v2) / (v1_norm * v2_norm)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Handle precision issues

        # Calculate angle in radians
        angle = torch.arccos(cos_theta)
        angles.append(angle.item())

    # Smooth angles to reduce noise
    angles = gaussian_filter(angles, sigma=5)

    # Find significant peaks (turns)
    peaks, _ = find_peaks(angles, height=threshold, distance=5)
    peaks_values = [angles[i] for i in peaks]

    # Ensure maximum angle is counted even if not detected as peak
    if angles:  # Only if there are angles to process
        max_angle = max(angles)
        if max_angle > threshold and max_angle not in peaks_values:
            peaks_values.append(max_angle)

    return peaks_values, len(peaks_values)


def dynamic_ratio(masks):
    # Downsample by taking every 5th frame to reduce computation
    masks = masks[::5, :, :]

    # Calculate total dynamic pixels and total pixels
    dynamic_pixels = torch.sum(masks)
    total_pixels = masks.shape[1] * masks.shape[2] * masks.shape[0]  # height * width * frames

    # Return ratio as float
    return (dynamic_pixels / total_pixels).item()


def process_single_row(row, index, args, device):
    video_id = row['id']
    rec_path = os.path.join(args.dir_path, video_id, "reconstructions")
    cam_pos_file = os.path.join(rec_path, "poses.npy")
    mask_file = os.path.join(rec_path, "dyn_masks.npz")

    # Check if required files exist
    if not os.path.exists(cam_pos_file) or not os.path.exists(mask_file):
        print(f"File not found: {cam_pos_file} or {mask_file}")
        return False, False, -1, -1, -1, -1, -1

    # Load camera data
    cam_pos, cam_rotate, time_steps, masks = load_file(cam_pos_file, mask_file, device)

    # Detect anomalies in trajectory
    anomaly = anomaly_detection(cam_pos, time_steps, args.anomaly_threshold, device)

    # Calculate movement metrics
    move_dist, dist_level = move_distance(cam_pos, time_steps, device)

    # Calculate total rotation
    rot_angle = rotation_angle(cam_rotate, time_steps, device)

    # Count trajectory turns
    traj_turns = trajectory_turns(cam_pos, time_steps, device)

    # Calculate dynamic object ratio
    dyn_ratio = dynamic_ratio(masks)

    return True, anomaly, move_dist, dist_level, rot_angle, traj_turns, dyn_ratio


def worker(task_queue, result_queue, args, worker_id):
    # Assign GPU based on worker ID if available
    device = torch.device(
        f"cuda:{args.gpu_id[worker_id % args.gpu_num]}"
        if torch.cuda.is_available() else "cpu"
    )

    while True:
        try:
            # Get next task with timeout
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break  # No more tasks

        # Process the task
        result = process_single_row(row, index, args, device)
        result_queue.put((index, result))
        task_queue.task_done()


def parse_args():
    parser = argparse.ArgumentParser(description="Camera Trajectory Evaluation")
    parser.add_argument("csv_path", type=str, help="Path to input CSV file containing video IDs")
    parser.add_argument("--dir_path", type=str, default="./outputs",
                        help="Base directory containing reconstruction data")
    parser.add_argument("--output_path", type=str, default="./outputs/evaluation_results.csv",
                        help="Path to save evaluation results")
    parser.add_argument("--anomaly_threshold", type=int, default=2,
                        help="Threshold for anomaly detection (consecutive errors)")
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1")')
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for parallel processing")
    parser.add_argument("--disable_parallel", action="store_true",
                        help="Disable parallel processing and run sequentially")
    return parser.parse_args()


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn')
    args = parse_args()

    # Process GPU IDs
    args.gpu_id = [int(gpu) for gpu in args.gpu_id.split(',')]
    args.gpu_num = len(args.gpu_id)

    # Load input CSV file
    df = pd.read_csv(args.csv_path)

    results = []
    if args.disable_parallel:
        # Process sequentially
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing videos"):
            device = torch.device(
                f"cuda:{args.gpu_id[0]}" if torch.cuda.is_available() else "cpu"
            )
            result = process_single_row(row, index, args, device)
            results.append((index, result))
    else:
        # Process in parallel
        manager = Manager()
        task_queue = manager.Queue()

        # Populate task queue
        for index, row in df.iterrows():
            task_queue.put((index, row))

        result_queue = manager.Queue()

        # Start worker processes
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for worker_id in range(args.num_workers):
                futures.append(executor.submit(worker, task_queue, result_queue, args, worker_id))

            # Wait for all workers to complete
            for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Workers completing"
            ):
                future.result()

        # Collect results from queue
        while not result_queue.empty():
            index, result = result_queue.get()
            results.append((index, result))

    # Sort results by original index
    results.sort(key=lambda x: x[0])

    # Add results to DataFrame
    df['success'] = [result[1][0] for result in results]
    df['anomaly'] = [result[1][1] for result in results]
    df['moveDist'] = [result[1][2] for result in results]
    df['distLevel'] = [result[1][3] for result in results]
    df['rotAngle'] = [result[1][4] for result in results]
    df['trajTurns'] = [result[1][5] for result in results]
    df['dynamicRatio'] = [result[1][6] for result in results]

    # Save results to CSV
    df.to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")
