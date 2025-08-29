"""
Batch inference script for dynamic mask generation using SAM2.
Processes video frames to generate dynamic object masks based on motion probabilities.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
import cv2
from scipy import ndimage
from scipy.sparse import csr_matrix
import argparse
import pandas as pd
import subprocess
from multiprocessing import Manager
import queue
import concurrent.futures
from tqdm import tqdm

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def compress(dyn_masks, save_path=None):
    """Compress dynamic masks using sparse matrix representation."""
    assert save_path.endswith(".npz")
    # Transform to sparse matrices
    sparse_matrices_list = [csr_matrix(dyn_mask) for dyn_mask in dyn_masks]
    sparse_matrices = {}
    for i, dyn_mask in enumerate(sparse_matrices_list):
        sparse_matrices[f"f_{i}_data"] = dyn_mask.data
        sparse_matrices[f"f_{i}_indices"] = dyn_mask.indices
        sparse_matrices[f"f_{i}_indptr"] = dyn_mask.indptr
        if i == 0:
            sparse_matrices["shape"] = dyn_mask.shape
    np.savez_compressed(save_path, **sparse_matrices)


def segment_sky(image):
    """Segment sky regions from image using HSV color space and morphological operations."""
    # Convert RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define range for blue color and create mask
    lower_blue = np.array([0, 0, 100])
    upper_blue = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue).view(bool)

    # Add luminous gray regions (likely sky)
    mask |= (hsv[:, :, 1] < 10) & (hsv[:, :, 2] > 150)
    mask |= (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 180)
    mask |= (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 220)

    # Morphological operations to clean up mask
    kernel = np.ones((5, 5), np.uint8)
    mask2 = ndimage.binary_opening(mask, structure=kernel)

    # Keep only largest connected components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask2.view(np.uint8), connectivity=8
    )
    cc_sizes = stats[1:, cv2.CC_STAT_AREA]
    order = cc_sizes.argsort()[::-1]  # Bigger first
    i = 0
    selection = []
    while i < len(order) and cc_sizes[order[i]] > cc_sizes[order[0]] / 2:
        selection.append(1 + order[i])
        i += 1
    mask3 = np.isin(labels, selection).reshape(labels.shape)

    # Return as tensor
    return torch.from_numpy(mask3)


def predict_mask(predictor, row, args, device):
    """Generate dynamic masks for a video using SAM2 and motion probabilities."""
    dir_path = os.path.join(args.dir_path, str(row["id"]))
    if not os.path.exists(dir_path):
        print(f"Directory '{dir_path}' not found. Exit.")
        return
    img_dir = os.path.join(args.dir_path, str(row["id"]), "img")
    if not os.path.exists(img_dir):
        print(f"Image directory '{img_dir}' not found. Exit.")
        return
    rec_dir = os.path.join(dir_path, "reconstructions")
    if not os.path.exists(rec_dir):
        print(f"Reconstructions directory '{rec_dir}' not found. Exit.")
        return
    prob_file = os.path.join(rec_dir, "motion_prob.npy")
    if not os.path.exists(prob_file):
        print(f"Motion probability file '{prob_file}' not found. Exit.")
        return

    compress_file = os.path.join(rec_dir, "dyn_masks.npz")
    if os.path.exists(compress_file):
        return

    # Load motion probabilities
    motion_probs = torch.from_numpy(np.load(prob_file)).to(device)

    # Load images
    images_list = list(sorted(glob(os.path.join(img_dir, "*.jpg"))))
    images = [cv2.imread(img_path) for img_path in images_list]

    if len(images) == 0 or len(images) != len(motion_probs):
        print(
            f"{row['video_path']},Number of frames ({len(images)}) does not match number of motion probabilities ({len(motion_probs)}). Exit."
        )
        return

    width, height = images[0].shape[1], images[0].shape[0]
    area = width * height

    masks = []
    # Process each frame
    for i in range(len(images)):
        motion_prob = motion_probs[i].to(device)

        # Segment sky to avoid false detections
        sky_mask = segment_sky(images[i])

        predictor.set_image(images[i])

        # Adaptive thresholding based on motion probability distribution
        prob_min, prob_max = motion_prob.min(), motion_prob.max()
        threshold = (prob_max - prob_min) * 0.4 + prob_min
        if threshold > prob_max - 0.1:
            masks.append(np.zeros((height, width), dtype=np.uint8))
            continue

        # Create initial mask from motion probabilities
        mask = (motion_prob < threshold).float()
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Find contours and use them as SAM2 prompts
        mask_np = mask.cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        merged_mask = np.zeros_like(mask_np)
        for c in contours:
            points = []
            for point in c:
                points.append(point[0])
            points = np.array(points)

            # Sample points from contour as prompts
            interval = len(points) // 3
            input_points = points[::interval].astype(np.float32)

            # Skip if points are in sky region
            if sky_mask[input_points[:, 1], input_points[:, 0]].any():
                continue

            input_labels = np.ones(input_points.shape[0], dtype=np.int64)

            # Use SAM2 to refine mask
            mask, score, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )

            # Skip if mask area is too large (likely background)
            if mask[0].sum() > area * 0.3:
                continue

            merged_mask = np.logical_or(merged_mask, mask[0])

        masks.append(merged_mask)

    # Save compressed masks
    masks = np.stack(masks, axis=0)
    compress(masks, compress_file)


def worker(task_queue, args, id):
    """Worker function for parallel dynamic mask generation."""
    gpu_id = id % args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Bind to specific GPU
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    sam2_model = None
    predictor = None

    while True:
        try:
            index, row = task_queue.get_nowait()
        except queue.Empty:
            break

        # Initialize SAM2 model and predictor lazily
        if sam2_model is None:
            sam2_model = build_sam2(
                args.model_cfg, args.checkpoints_path, device=device
            )
        if predictor is None:
            predictor = SAM2ImagePredictor(sam2_model)

        predictor.reset_predictor()
        predict_mask(predictor, row, args, device)


def parse_args():
    """Parse command line arguments for dynamic mask generation."""
    parser = argparse.ArgumentParser(description="SAM2 Image Predictor")
    parser.add_argument("csv_path", type=str, help="Path to the csv file")
    parser.add_argument(
        "--dir_path",
        type=str,
        required=True,
        help="Path to the directory containing images and masks",
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="#workers for concurrent.futures"
    )
    parser.add_argument("--gpu_num", type=int, default=1, help="gpu number")
    parser.add_argument(
        "--checkpoints_path",
        type=str,
        default="checkpoints",
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="Path to the model configuration file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.csv_path):
        print(f"Meta file '{args.csv_path}' not found. Exit.")
        return

    # Set SAM2 checkpoint path
    args.checkpoints_path = os.path.join(
        args.checkpoints_path, "SAM2/sam2.1_hiera_large.pt"
    )

    df = pd.read_csv(args.csv_path)

    # Setup multiprocessing
    manager = Manager()
    task_queue = manager.Queue()
    for index, row in df.iterrows():
        task_queue.put((index, row))

    # Process tasks with multiple workers
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        futures = []
        for id in range(args.num_workers):
            futures.append(executor.submit(worker, task_queue, args, id))

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Finished workers",
        ):
            future.result()


if __name__ == "__main__":
    main()
