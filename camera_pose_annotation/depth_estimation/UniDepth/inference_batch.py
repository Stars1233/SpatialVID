"""
Distributed batch inference script for UniDepth V2 model.
Processes video frames to generate depth maps and camera intrinsics using distributed computing.
"""

import argparse
from datetime import timedelta
import glob
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from unidepth.models import UniDepthV2


class ImageDataset(Dataset):
    """Dataset for loading and preprocessing images for UniDepth inference."""

    def __init__(self, img_list, input_size):
        self.img_list = img_list
        self.input_size = input_size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        """Load and preprocess a single image with error handling."""

        def inner_func(idx):
            img_path = self.img_list[idx]
            rgb = np.array(Image.open(img_path))[..., :3]

            h, w = rgb.shape[:2]

            # Calculate target size maintaining aspect ratio
            if w > h:
                final_w, final_h = self.input_size, int(round(self.input_size * h / w))
            else:
                final_w, final_h = int(round(self.input_size * w / h)), self.input_size

            rgb_resized = cv2.resize(rgb, (final_w, final_h), cv2.INTER_AREA)
            rgb_torch = (
                torch.from_numpy(rgb_resized).permute(2, 0, 1).float()
            )  # Convert to CHW format

            return {
                "image": rgb_torch,
                "path": img_path,
            }

        while True:
            try:
                return inner_func(idx)
            except Exception as e:
                print(f"e: [{e}], path: {self.img_list[idx]}, try to get next idx")
                idx = (idx + 1) % len(self.img_list)
                if idx >= len(self.img_list):
                    raise StopIteration


def collate_fn(batch):
    """Custom collate function for batching data."""
    return_batch = {}
    for key in batch[0].keys():
        if key == "image":
            return_batch[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            return_batch[key] = [item[key] for item in batch]
    return return_batch


def parse_args():
    """Parse command line arguments for UniDepth inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, help="Path to the csv file")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--checkpoints_path", type=str, default="./checkpoints")
    parser.add_argument(
        "--input_size", type=int, default=640, help="Input size for the model"
    )
    parser.add_argument("--bs", type=int, default=8, help="Inference batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Data loading workers"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize distributed environment
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    DEVICE = f"cuda:{local_rank}"

    # Load data list from CSV
    df = pd.read_csv(args.csv_path)

    img_list = []
    for index, row in tqdm(
        df.iterrows(), total=len(df), desc="Loading images", disable=local_rank != 0
    ):
        img_dir = os.path.join(args.output_dir, row["id"], "img")
        if not os.path.exists(img_dir):
            print(f"Image directory not found: {img_dir}")
            continue
        img_list += sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        img_list += sorted(glob.glob(os.path.join(img_dir, "*.png")))

    # Create dataset and distributed sampler
    dataset = ImageDataset(img_list, args.input_size)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=local_rank,
        shuffle=False,
        drop_last=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Initialize UniDepth V2 model
    load_from = os.path.join(args.checkpoints_path, "UniDepth")
    model = UniDepthV2.from_pretrained(load_from)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Run inference and save results
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc="Processing batches", disable=(local_rank != 0)
        ):
            images = batch["image"].to(device)
            paths = batch["path"]

            # Model inference
            predictions = model.infer(images)

            # Process results for each sample
            for i in range(len(paths)):
                depth = predictions["depth"][i, 0].cpu().numpy()  # [H, W]
                intrinsics = predictions["intrinsics"][i].cpu().numpy()
                focal_length = intrinsics[
                    0, 0
                ]  # Assume principal point at center, take fx
                w = depth.shape[-1]  # Width

                # Calculate FOV (horizontal field of view)
                fov = np.rad2deg(2 * np.arctan(w / (2 * focal_length)))

                # Save results
                img_path = paths[i]
                output_filename = (
                    os.path.splitext(os.path.basename(img_path))[0] + ".npz"
                )
                output_dir = os.path.join(
                    os.path.dirname(os.path.dirname(img_path)), "unidepth"
                )
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, output_filename)
                np.savez(output_path, depth=np.float32(depth), fov=fov)


if __name__ == "__main__":
    main()
