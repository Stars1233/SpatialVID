"""
Distributed batch inference script for Depth-Anything V2 model.
Processes video frames to generate depth maps using distributed computing.
"""

import argparse
from datetime import timedelta
import cv2
import glob
import numpy as np
import pandas as pd
import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision.transforms import Compose
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from tqdm import tqdm

from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from depth_anything_v2.dpt import DepthAnythingV2

# Model configuration for different encoder variants
model_configs = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536],
    },
}


class ImageDataset(Dataset):
    """Dataset for loading and preprocessing images for depth estimation."""

    def __init__(self, img_list, input_size):
        self.img_list = img_list
        self.input_size = input_size
        self.transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )

    def __len__(self):
        return len(self.img_list)

    def image2tensor(self, raw_image):
        """Convert raw image to tensor format for model input."""
        h, w = raw_image.shape[:2]

        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0

        image = self.transform({"image": image})["image"]
        image = torch.from_numpy(image)

        return image, (h, w)

    def __getitem__(self, idx):
        """Load and preprocess a single image with error handling."""

        def inner_func(idx):
            img_path = self.img_list[idx]
            raw_image = cv2.imread(img_path)
            height, width = raw_image.shape[:2]
            if height != 720 or width != 1280:
                raise ValueError(f"Image size is not 720x1280, but {height}x{width}")

            image, (original_h, original_w) = self.image2tensor(raw_image)

            data = {
                "image": image,
                "path": img_path,
                "original_size": (original_h, original_w),
            }
            return data

        while True:
            try:
                return inner_func(idx)
            except Exception as e:
                print(f"e: [{e}], path: {self.img_list[idx]}, try to get next idx")
                idx += 1
                if idx >= len(self.img_list):
                    raise StopIteration


def parse_args():
    """Parse command line arguments for depth estimation."""
    parser = argparse.ArgumentParser(
        description="Depth Anything V2 Distributed Inference"
    )
    parser.add_argument("csv_path", type=str, help="Path to the csv file")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--dir_path", type=str, default="./vis_depth")
    parser.add_argument(
        "--encoder", type=str, default="vitl", choices=["vits", "vitb", "vitl", "vitg"]
    )
    parser.add_argument("--checkpoints_path", type=str, default="./checkpoints")
    parser.add_argument("--bs", type=int, default=8, help="Batch size for inference")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )
    return parser.parse_args()


def collate_fn(batch):
    """Custom collate function for batching data."""
    return_batch = {}
    for key in batch[0].keys():
        if key == "image":
            return_batch[key] = torch.stack([item[key] for item in batch], dim=0)
        else:
            return_batch[key] = [item[key] for item in batch]
    return return_batch


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
        df.iterrows(), total=len(df), desc="Loading images", disable=(local_rank != 0)
    ):
        img_dir = os.path.join(args.dir_path, row["id"], "img")
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

    # Initialize Depth-Anything V2 model
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    load_from = os.path.join(
        args.checkpoints_path, f"Depth-Anything/depth_anything_v2_{args.encoder}.pth"
    )
    depth_anything.load_state_dict(torch.load(load_from, map_location="cpu"))
    depth_anything = depth_anything.to(DEVICE).eval()

    # Run inference and save depth maps
    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc="Depth inference", disable=(local_rank != 0)
        ):
            images = batch["image"].to(DEVICE)
            original_sizes = batch["original_size"]
            paths = batch["path"]

            # Forward pass through depth model
            depth = depth_anything(images)

            # Upsample to original image size
            original_h, original_w = original_sizes[0]
            depth = F.interpolate(
                depth[:, None],
                size=(original_h, original_w),
                mode="bilinear",
                align_corners=False,
            )

            # Save depth maps as numpy arrays
            for i in range(depth.shape[0]):
                depth_i = depth[i, 0].cpu().numpy()
                img_path = paths[i]
                output_filename = (
                    os.path.splitext(os.path.basename(img_path))[0] + ".npy"
                )
                output_dir = os.path.join(
                    os.path.dirname(os.path.dirname(img_path)), "depth-anything"
                )
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, output_filename)
                np.save(output_path, depth_i)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
