import argparse
import os

import colossalai
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from mmengine import Config
from mmengine.dataset import Compose, default_collate
from mmengine.registry import DefaultScope
from mmocr.datasets import PackTextDetInputs # type: ignore
from mmocr.registry import MODELS # type: ignore
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import CenterCrop, Compose, Resize
from tqdm import tqdm

# add tools to path
# import sys
# path = '/mnt/e/vid_anno_step1'
# sys.path.append(path)
from tools.utils.load_frames import load_frames

CFG_PATH = "tools/scoring/ocr/dbnetpp.py"

def merge_scores(gathered_list: list, meta: pd.DataFrame):
    # 收集所有结果
    indices_list = []
    scores_list = []
    for sublist in gathered_list:
        indices_part, scores_part = sublist
        indices_list.extend(indices_part)
        scores_list.extend(scores_part)
    
    # 创建DataFrame进行分组平均
    df = pd.DataFrame({
        "original_index": indices_list,
        "scores": scores_list
    })
    mean_scores = df.groupby("original_index")["scores"].mean().reset_index()
    
    # 更新meta数据
    meta.loc[mean_scores["original_index"], "ocr"] = mean_scores["scores"].values


class VideoTextDataset(torch.utils.data.Dataset):
    def __init__(self, meta_path, transform, fig_load_dir):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.transform = transform
        self.transform = Compose(
            [
                Resize(1024),
                CenterCrop(1024),
            ]
        )
        self.formatting = PackTextDetInputs(meta_keys=["scale_factor"])
        self.fig_load_dir = fig_load_dir

    def __len__(self):
        return len(self.meta) * 3  # 数据集长度翻倍

    def __getitem__(self, index):
        # 计算原始视频索引和帧索引
        video_idx = index // 3
        group_idx = index % 3
        
        row = self.meta.iloc[video_idx]

        # 提取特定帧
        img = load_frames(
            video_path=row["video_path"],
            fig_load_dir=self.fig_load_dir,
            id_ori=row["id_ori"],
            id=row["id"],
            usage="ocr",
            group_idx=group_idx
        )

        img = self.transform(img)
        img_array = np.array(img)[:, :, ::-1].copy()
        results = {
            "img": img_array,
            "scale_factor": 1.0,
        }
        results = self.formatting(results)
        results["original_index"] = video_idx  # 保存原始索引

        return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers")
    parser.add_argument("--fig_load_dir", type=str, required=True, help="Directory to load the extracted frames")
    parser.add_argument("--skip_if_existing", action="store_true")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_ocr{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    cfg = Config.fromfile(CFG_PATH)
    colossalai.launch_from_torch({})

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    DefaultScope.get_instance("ocr", scope_name="mmocr")  # use mmocr Registry as default

    # build model
    model = MODELS.build(cfg.model)
    model.init_weights()
    model.to(device)  # set data_preprocessor._device
    print("==> Model built.")

    # build dataset
    transform = Compose(cfg.test_pipeline)
    dataset = VideoTextDataset(meta_path=meta_path, transform=transform, fig_load_dir=args.fig_load_dir)
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
        collate_fn=default_collate,
    )
    print("==> Dataloader built.")

    # compute scores
    dataset.meta["ocr"] = np.nan
    indices_list = []
    scores_list = []
    model.eval()
    for data in tqdm(dataloader, disable=(dist.get_rank() != 0), position=dist.get_rank()):
        indices_i = data["original_index"]  # 修改这里使用original_index
        indices_list.extend(indices_i.tolist())
        del data["original_index"]

        pred = model.test_step(data)  # this line will cast data to device

        num_texts_i = [(x.pred_instances.scores > 0.3).sum().item() for x in pred]
        scores_list.extend(num_texts_i)

    gathered_list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_list, (indices_list, scores_list))

    if dist.get_rank() == 0:
        merge_scores(gathered_list, dataset.meta)
        dataset.meta.to_csv(out_path, index=False)
        print(f"New meta (shape={dataset.meta.shape}) with ocr results saved to '{out_path}'.")


if __name__ == "__main__":
    main()