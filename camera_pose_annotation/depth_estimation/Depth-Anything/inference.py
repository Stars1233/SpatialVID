import argparse
import cv2
import glob
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2
    
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}


def parse_args():
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--dir_path', type=str, default='./vis_depth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/Depth-Anything/depth_anything_v2_vitl.pth')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    img_path = os.path.join(args.dir_path, "img")
    out_path = os.path.join(args.dir_path, "depth-anything")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    img_list = sorted(glob.glob(os.path.join(img_path, "*.jpg")))
    img_list += sorted(glob.glob(os.path.join(img_path, "*.png")))

    for k, img in enumerate(img_list):
        print(f'Progress {k+1}/{len(img_list)}: {img}')
        
        raw_image = cv2.imread(img)
        
        depth = depth_anything.infer_image(raw_image, args.input_size)

        output_path = os.path.join(out_path, os.path.splitext(os.path.basename(img))[0] + '.npy')
        np.save(output_path, depth)