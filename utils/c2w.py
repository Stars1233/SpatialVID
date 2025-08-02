import einops
import torch
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import argparse
import concurrent.futures
import multiprocessing as mp
from multiprocessing import Manager
import queue
from tqdm import tqdm

import shutil
import glob

class Pose():
    """
    A class of operations on camera poses (numpy arrays with shape [...,3,4]).
    Each [3,4] camera pose takes the form of [R|t].
    """

    def __call__(self, R=None, t=None):
        """
        Construct a camera pose from the given rotation matrix R and/or translation vector t.

        Args:
            R: Rotation matrix [...,3,3] or None
            t: Translation vector [...,3] or None

        Returns:
            pose: Camera pose matrix [...,3,4]
        """
        assert R is not None or t is not None
        if R is None:
            if not isinstance(t, np.ndarray):
                t = np.array(t)
            R = np.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, np.ndarray):
                R = np.array(R)
            t = np.zeros(R.shape[:-1], device=R.device)
        else:
            if not isinstance(R, np.ndarray):
                R = np.array(R)
            if not isinstance(t, np.ndarray):
                t = np.tensor(t)
        assert R.shape[:-1] == t.shape and R.shape[-2:] == (3, 3)
        R = R.astype(np.float32)
        t = t.astype(np.float32)
        pose = np.concatenate([R, t[..., None]], axis=-1)  # [...,3,4]
        assert pose.shape[-2:] == (3, 4)
        return pose

    def invert(self, pose, use_inverse=False): # c2w <==> w2c
        """
        Invert a camera pose.

        Args:
            pose: Camera pose matrix [...,3,4]
            use_inverse: Whether to use matrix inverse instead of transpose

        Returns:
            pose_inv: Inverted camera pose matrix [...,3,4]
        """
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(0, 2, 1)
        t_inv = (-R_inv @ t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        """
        Compose a sequence of poses together.
        pose_new(x) = poseN o ... o pose2 o pose1(x)

        Args:
            pose_list: List of camera poses to compose

        Returns:
            pose_new: Composed camera pose
        """
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        """
        Compose two poses together.
        pose_new(x) = pose_b o pose_a(x)

        Args:
            pose_a: First camera pose
            pose_b: Second camera pose

        Returns:
            pose_new: Composed camera pose
        """
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b @ R_a
        t_new = (R_b @ t_a + t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new

    def scale_center(self, pose, scale):
        """
        Scale the camera center from the origin.
        0 = R@c+t --> c = -R^T@t (camera center in world coordinates)
        0 = R@(sc)+t' --> t' = -R@(sc) = -R@(-R^T@st) = st

        Args:
            pose: Camera pose to scale
            scale: Scale factor

        Returns:
            pose_new: Scaled camera pose
        """
        R, t = pose[..., :3], pose[..., 3:]
        pose_new = np.concatenate([R, t * scale], axis=-1)
        return pose_new

def quaternion_to_matrix(quaternions, eps: float = 1e-8):
    '''
    Convert the 4-dimensional quaternions to 3x3 rotation matrices.
    This is adapted from Pytorch3D:
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
    '''

    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return einops.rearrange(o, "... (i j) -> ... i j", i=3, j=3)

def pose_from_quaternion(pose):  # input is w2c, pose(n,7) or (n,v,7), after_pose (N,3,4) after_pose: w2c
    # tensor in https://github.com/pointrix-project/Geomotion/blob/6ab0c364f1b44ab4ea190085dbf068f62b42727c/geomotion/model/cameras.py#L6
    if type(pose) == np.ndarray:
        pose = torch.tensor(pose)
    if len(pose.shape) == 1:
        pose = pose[None]
    quat_t = pose[..., :3]
    quat_r = pose[..., 3:]
    w2c_matrix = torch.zeros((*list(pose.shape)[:-1],3,4), device=pose.device)
    w2c_matrix[..., :3, 3] = quat_t
    w2c_matrix[..., :3, :3] = quaternion_to_matrix(quat_r)
    return w2c_matrix


def possess_single_row(row, index, args):
    id = row['id']
    dir_path = os.path.join(args.dir_path, id, "reconstructions")
    cam_pos_file = os.path.join(dir_path, "poses.npy")
    if not os.path.exists(cam_pos_file):
        return
    extrinsic_file = os.path.join(dir_path, "extrinsics.npy")
    if os.path.exists(extrinsic_file):
        return
    
    pose = np.load(cam_pos_file)
    # w2c: (N,v,7) --> w2c: (N,v,3,4)
    poses = pose_from_quaternion(pose) # pose: (N,7), after_pose (N,3,4) after_pose: w2c
    poses = poses.cpu().numpy()
    # w2c: (N,v,3,4) --> c2w: (N,v,3,4)
    pose_inv = Pose().invert(poses)
    np.save(extrinsic_file, pose_inv)
    
        

def worker(task_queue, args, pbar):
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        possess_single_row(row, index, args)
        task_queue.task_done()
        pbar.update(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert quaternion to camera pose")
    parser.add_argument("csv_path", type=str, help="Path to the csv file")
    parser.add_argument("--dir_path", type=str, default="./outputs")
    parser.add_argument("--num_workers", type=int,
                        default=8, help="Number of workers for parallel processing")
    parser.add_argument("--disable_parallel", action="store_true",
                        help="Disable parallel processing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    df = pd.read_csv(args.csv_path)
    
    if args.disable_parallel:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            possess_single_row(row, index, args)
    else:
        manager = Manager()
        task_queue = manager.Queue()
        for index, row in df.iterrows():
            task_queue.put((index, row))
            
        with tqdm(total=len(df), desc="Finished tasks") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                futures = []
                for _ in range(args.num_workers):
                    futures.append(executor.submit(worker, task_queue, args, pbar))
                for future in concurrent.futures.as_completed(futures):
                    future.result()
