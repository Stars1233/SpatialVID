import cv2  # isort:skip

import argparse
import gc
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

# import sys
# open_sora_path = '/home/wjh/video_anno/reference_code/open_sora'
# sys.path.append(open_sora_path)

from tools.utils.load_frames import load_frames
from tools.scoring.optical_flow.unimatch import UniMatch

torch.backends.cudnn.enabled = False # This line enables large batch, but the speed is similar


def merge_scores(gathered_list: list, meta: pd.DataFrame, column):
    # 解包所有结果
    video_indices = []
    scores = []
    for sublist in gathered_list:
        indices_part, scores_part = sublist
        video_indices.extend(indices_part)
        scores.extend(scores_part)
    
    # 创建统计DataFrame
    df = pd.DataFrame({
        "video_index": video_indices,
        "score": scores
    })
    
    # 分组计算统计量
    grouped = df.groupby("video_index")["score"]
    meta[f"{column}_mean"] = grouped.transform("mean")
    meta[f"{column}_max"] = grouped.transform("max")
    meta[f"{column}_min"] = grouped.transform("min")
    
    return meta


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, fig_load_dir):  # 改为接受多组帧
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.fig_load_dir = fig_load_dir

    def __len__(self):
        return len(self.meta) * 3  # 样本数 = 视频数 × 帧组数

    def __getitem__(self, index):
        # 计算原始视频索引和帧组索引
        video_idx = index // 3
        group_idx = index % 3
        
        sample = self.meta.iloc[video_idx]

        # 提取当前组的帧
        images = load_frames(
            video_path=sample["video_path"], 
            fig_load_dir=self.fig_load_dir,
            id_ori=sample["id_ori"],
            id=sample["id"],
            usage="flow",
            group_idx= group_idx
        )

        # transform处理
        images = torch.stack([pil_to_tensor(x) for x in images])
        images = images.float()
        H, W = images.shape[-2:]
        if H > W:
            images = rearrange(images, "N C H W -> N C W H")
        images = F.interpolate(images, size=(320, 576), mode="bilinear", align_corners=True)

        return {
            "video_index": video_idx,  # 原始视频索引
            "group_index": group_idx,  # 帧组索引
            "images": images
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--bs", type=int, default=4, help="Batch size")  # don't use too large bs for unimatch
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--fig_load_dir", type=str, required=True, help="Directory to load the extracted frames")
    parser.add_argument("--skip_if_existing", action="store_true")
    parser.add_argument("--use_cudnn", action="store_true", help="Use CuDNN")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    use_cudnn = args.use_cudnn
    print(f"Use CuDNN: {use_cudnn}")
    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_flow{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    # build model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = UniMatch(
        feature_channels=128,
        num_scales=2,
        upsample_factor=4,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        reg_refine=True,
        task="flow",
    )
    ckpt = torch.load("./pretrained_models/unimatch/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth")
    model.load_state_dict(ckpt["model"])
    model = model.to(device)

    # build dataset
    dataset = VideoTextDataset(meta_path=meta_path, fig_load_dir=args.fig_load_dir)
    # print(f"Dataset size: {len(dataset)}") # 142866
    dataloader = DataLoader(
        dataset,
        batch_size=args.bs,
        num_workers=args.num_workers,
        sampler=DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False,
            drop_last=False,
        ),
    )

    # compute optical flow scores
    indices_list = []
    scores_list = []
    model.eval()
    print('start in')
    for batch in tqdm(dataloader, disable=dist.get_rank() != 0):
        video_indices = batch["video_index"]
        images = batch["images"].to(device, non_blocking=True)

        B = images.shape[0]
        batch_0 = rearrange(images[:, :-1], "B N C H W -> (B N) C H W").contiguous()
        batch_1 = rearrange(images[:, 1:], "B N C H W -> (B N) C H W").contiguous()

        with torch.no_grad():
            res = model(
                batch_0,
                batch_1,
                attn_type="swin",
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
                num_reg_refine=6,
                task="flow",
                pred_bidir_flow=False,
            )
            flow_maps = res["flow_preds"][-1].cpu()  # [B * (N-1), 2, H, W]
            flow_maps = rearrange(flow_maps, "(B N) C H W -> B N H W C", B=B)
            flow_scores = flow_maps.abs().mean(dim=[1, 2, 3, 4]) # 
            flow_scores = flow_scores.tolist()

        indices_list.extend(video_indices.tolist())
        scores_list.extend(flow_scores)

    # wait for all ranks to finish data processing
    dist.barrier()

    torch.cuda.empty_cache()
    gc.collect()
    gathered_list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_list, (indices_list, scores_list))
    if dist.get_rank() == 0:
        meta_new = merge_scores(gathered_list, dataset.meta, column="flow")
        meta_new.to_csv(out_path, index=False)
        print(f"New meta with optical flow scores saved to '{out_path}'.")


if __name__ == "__main__":
    main()