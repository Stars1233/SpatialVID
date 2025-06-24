import argparse
import os
import gc
from glob import glob
from datetime import timedelta
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import cv2
import cpbd
from tqdm import tqdm


def compute_blur(image):
    """
    Compute blur amount using CPBD (Cumulative Probability of Blur Detection).
    """
    return cpbd.compute(image)


def merge_scores(gathered_list: list, meta: pd.DataFrame, column):
    # reorder
    indices_list = list(map(lambda x: x[0], gathered_list))
    scores_list = list(map(lambda x: x[1], gathered_list))

    flat_indices = []
    for x in zip(*indices_list):
        flat_indices.extend(x)
    flat_scores = []
    for x in zip(*scores_list):
        flat_scores.extend(x)
    flat_indices = np.array(flat_indices)
    flat_scores = np.array(flat_scores)

    # filter duplicates
    unique_indices, unique_indices_idx = np.unique(flat_indices, return_index=True)
    meta.loc[unique_indices, column] = flat_scores[unique_indices_idx]

    # drop indices in meta not in unique_indices
    meta = meta.loc[unique_indices]
    return meta


class VideoDataset(torch.utils.data.Dataset):
    """Dataset to handle video blur computation."""
    def __init__(self, meta_path, fig_load_dir):
        self.meta = pd.read_csv(meta_path)
        self.fig_load_dir = fig_load_dir
    
    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        
        # load images
        images_dir = os.path.join(self.fig_load_dir, sample["id"])
        images = sorted(glob(f"{images_dir}/img/*.jpg"))[:3]
        
        # Compute blur amount
        blur_scores = []
        for img in images:
            frame = Image.open(img).convert("RGB")
            # convert to numpy
            frame = np.array(frame)
            # convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # compute blur amount
            blur_score = compute_blur(gray, use_cuda=True)
            blur_scores.append(blur_score)
        
        return {"index": index, "blur": np.mean(blur_scores)}
    
    def __len__(self):
        return len(self.meta)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--load_num", type=int, default=4, help="Number of frames to load")
    parser.add_argument("--bs", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--fig_load_dir", type=str, required=True, help="Directory to load the extracted frames")
    parser.add_argument("--skip_if_existing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    meta_path = args.meta_path
    
    if not os.path.exists(meta_path):
        print(f"Metadata file '{meta_path}' not found. Exiting.")
        return
    
    output_path = meta_path.replace(".csv", "_blur.csv")
    if args.skip_if_existing and os.path.exists(output_path):
        print(f"Output '{output_path}' already exists. Exiting.")
        return
    
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count()) if torch.cuda.is_available() else None
    
    dataset = VideoDataset(meta_path, fig_load_dir=args.fig_load_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        sampler=DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
    )
    
    indices_list, scores_list = [], []
    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0), position=dist.get_rank()):
        indices = batch["index"].cpu().numpy()
        scores_np = batch["blur"].cpu().numpy()
        
        indices_list.extend(indices.tolist())
        scores_list.extend(scores_np.tolist())

    # wait for all ranks to finish data processing
    dist.barrier()

    torch.cuda.empty_cache()
    gc.collect()
    gathered_list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_list, (indices_list, scores_list))
    if dist.get_rank() == 0:
        meta_new = merge_scores(gathered_list, dataset.meta, column="blur")
        meta_new.to_csv(output_path, index=False)
        print(f"New meta with luminance scores saved to '{output_path}'")
    

if __name__ == "__main__":
    main()
