"""
Single-threaded inference script for UniDepth V2 model.
Processes images in a directory to generate depth maps and camera parameters sequentially.
"""

import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image
import torch
from unidepth.models import UniDepthV2

# Maximum dimension for image resizing
LONG_DIM = 640


def parse_args():
    """Parse command line arguments for UniDepth inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, default="./vis_depth")
    parser.add_argument("--load-from", type=str, default="checkpoints/UniDepth")

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize UniDepth V2 model
    model = UniDepthV2.from_pretrained(args.load_from)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Setup input and output paths
    img_path = os.path.join(args.dir_path, "img")
    out_path = os.path.join(args.dir_path, "unidepth")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Collect all image files
    img_list = sorted(glob.glob(os.path.join(img_path, "*.jpg")))
    img_list += sorted(glob.glob(os.path.join(img_path, "*.png")))

    fovs = []
    # Process each image sequentially
    for img_path in img_list:
        # Load and preprocess image
        rgb = np.array(Image.open(img_path))[..., :3]

        # Calculate target size maintaining aspect ratio
        if rgb.shape[1] > rgb.shape[0]:
            final_w, final_h = LONG_DIM, int(
                round(LONG_DIM * rgb.shape[0] / rgb.shape[1])
            )
        else:
            final_w, final_h = (
                int(round(LONG_DIM * rgb.shape[1] / rgb.shape[0])),
                LONG_DIM,
            )
        rgb = cv2.resize(rgb, (final_w, final_h), cv2.INTER_AREA)

        # Convert to tensor format
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

        # Predict depth and intrinsics
        predictions = model.infer(rgb_torch)

        # Calculate FOV (horizontal field of view) from predicted intrinsics
        fov_ = np.rad2deg(
            2
            * np.arctan(
                predictions["depth"].shape[-1]
                / (2 * predictions["intrinsics"][0, 0, 0].cpu().numpy())
            )
        )
        depth = predictions["depth"][0, 0].cpu().numpy()
        print(fov_)
        fovs.append(fov_)

        # Save depth map and FOV
        np.savez(
            os.path.join(out_path, img_path.split("/")[-1][:-4] + ".npz"),
            depth=np.float32(depth),
            fov=fov_,
        )


if __name__ == "__main__":
    main()
