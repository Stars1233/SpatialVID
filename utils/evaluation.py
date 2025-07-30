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
    reconstructed_sparse_matrices = []
    num_frames = (len(loaded_data) - 1) // 3
    matrix_shape = loaded_data['shape']

    for i in range(num_frames):
        data = loaded_data[f'f_{i}_data']
        indices = loaded_data[f'f_{i}_indices']
        indptr = loaded_data[f'f_{i}_indptr']
        # reconstruction csr_matrix
        reconstructed_matrix = csr_matrix((data, indices, indptr), shape=matrix_shape)
        reconstructed_sparse_matrices.append(reconstructed_matrix)

    reconstructed_mask_3d = np.stack([m.toarray() for m in reconstructed_sparse_matrices], axis=0)
    # reconstructed_mask_3d.shape
    return reconstructed_mask_3d


def load_file(cam_pos_file, mask_file, device):
    try:
        prams = torch.from_numpy(np.load(cam_pos_file)).float().to(device)
        cam_pos = prams[:, :3]
        cam_rotate = prams[:, 3:]
        time_steps = prams.shape[0]
        masks = torch.from_numpy(expand(np.load(mask_file))).to(device)
    except FileNotFoundError:
        print(f"error: {cam_pos_file} not found")
        exit()
    except Exception as e:
        print(f"error occurred in {cam_pos_file}: {e}")
        exit()

    return cam_pos, cam_rotate, time_steps, masks


def anomaly_detection(cam_pos, time_steps, threshold, device):
    if time_steps < 4:
        return True
    preds = torch.zeros((time_steps, 3), dtype=torch.float32, device=device)
    # linear prediction
    err_count = 0
    for t in range(0, time_steps - 3):
        v1 = cam_pos[t + 2] - cam_pos[t + 1]
        v2 = cam_pos[t + 1] - cam_pos[t]
        a = v1 - v2
        preds[t + 3] = cam_pos[t + 2] + v1 + 0.5 * a
        error = torch.sqrt(torch.sum((preds[t + 3] - cam_pos[t + 3]) ** 2))

        if error > 0.03:
            err_count += 1
            if err_count >= threshold:
                return True
        else:
            err_count = 0

    return False


def motion_intensity(cam_pos, time_steps, device):
    total_distance = torch.tensor(0., dtype=torch.float32, device=device)
    high_thres = 2.41
    medium_thres = 0.92
    low_thres = 0.28
    very_low_thres = 0.08
    intensity = 0

    # 计算每帧之间的距离
    for i in range(0, time_steps - 1):
        total_distance += torch.norm(cam_pos[i + 1] - cam_pos[i])

    total_distance = total_distance.item()
    if total_distance >= high_thres:
        intensity = 4
    elif total_distance >= medium_thres:
        intensity = 3
    elif total_distance >= low_thres:
        intensity = 2
    elif total_distance >= very_low_thres:
        intensity = 1
    else:
        intensity = 0

    return total_distance, intensity


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]

    # 构建矩阵
    matrix = torch.tensor([
        [w1, -z1, y1, x1],
        [z1, w1, -x1, y1],
        [-y1, x1, w1, z1],
        [-x1, -y1, -z1, w1]
    ], dtype=q1.dtype, device=q1.device)

    # 构建向量
    vector = torch.tensor([x2, y2, z2, w2], dtype=q2.dtype, device=q2.device)

    # 矩阵乘法
    result = torch.matmul(matrix, vector)

    return result

def rotation_angle(cam_rotate, time_steps, device):
    total_rad = torch.tensor(0.0, device=device)
    for i in range(0, time_steps - 1):
        q1 = cam_rotate[i]
        q2 = cam_rotate[i + 1]
        q1_inverse = torch.stack([-q1[0], -q1[1], -q1[2], q1[3]], dim=0)
        q_relative = quaternion_multiply(q2, q1_inverse)
        w = q_relative[3]
        w = torch.clamp(w, -1.0, 1.0)  # 处理数值误差
        rotation_angle_rad = 2 * torch.arccos(w)
        total_rad += rotation_angle_rad

    return total_rad.item()


def arc_counter(cam_pos, time_steps, device, threshold=0.45):
    if time_steps < 3:  # 至少需要3个点才能形成两个向量
        return [], 0
    
    total_angle = torch.tensor(0.0, dtype=torch.float32, device=device)
    angles = []
    for t in range(1,time_steps - 1):
        v1 = cam_pos[t] - cam_pos[0]
        v2 = cam_pos[time_steps-1] - cam_pos[t]
        
        # 计算向量模长（避免除以0）
        v1_norm = torch.norm(v1)
        v2_norm = torch.norm(v2)
        if v1_norm < 1e-8 or v2_norm < 1e-8:
            continue  # 跳过长度为0的向量（避免NaN）
        
        # 点积计算夹角余弦值（限制在[-1,1]范围内防误差）
        cos_theta = torch.dot(v1, v2) / (v1_norm * v2_norm)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        
        # 计算夹角（弧度）并累加
        angle = torch.arccos(cos_theta)
        angles.append(angle.item())

    # 将angles平滑
    angles = gaussian_filter(angles, sigma=5)

    peaks, _ = find_peaks(angles, height=threshold, distance=5)
    peaks_values = [angles[i] for i in peaks]

    # 判断angles最大值是否在peaks中
    max_angle = max(angles)
    if max_angle > threshold and max_angle not in peaks_values:
        peaks_values.append(max_angle)
    
    return peaks_values, len(peaks_values)


def mask_ratio(masks):
    # 每隔5帧取一个mask
    masks = masks[::5, :, :]
    # 计算每一个mask占整个图像的占比
    mask_area = torch.sum(masks)  # (num_frames,)
    total_area = masks.shape[1] * masks.shape[2] * masks.shape[0]  # (1,)
    mask_area_ratio = mask_area / total_area  # (1,)
    return mask_area_ratio.item()


def possess_single_row(row, index, args, device):
    id = row['id']
    rec_path = os.path.join(args.dir_path, id, "reconstructions")
    cam_pos_file = os.path.join(rec_path, "poses.npy")
    mask_file = os.path.join(rec_path, "dyn_masks.npz")
    if not os.path.exists(cam_pos_file) or not os.path.exists(mask_file):
        print(f"File not found: {cam_pos_file} or {mask_file}")
        return False, False, -1, -1, -1, -1, -1, -1

    # 加载相机位姿数据
    cam_pos, cam_rotate, time_steps, masks = load_file(cam_pos_file, mask_file, device)

    # 检测异常
    anomaly = anomaly_detection(cam_pos, time_steps, args.anomaly_threshold, device)

    # 计算运动强度
    total_distance, intensity = motion_intensity(cam_pos, time_steps, device)
    
    # 计算旋转角度
    total_rotation_angle = rotation_angle(cam_rotate, time_steps, device)

    # 计算弧形个数
    arcs, arcs_num = arc_counter(cam_pos, time_steps, device)

    # 计算动态mask占比
    mask_area_ratio = mask_ratio(masks)

    return True, anomaly, total_distance, intensity, total_rotation_angle, arcs, arcs_num, mask_area_ratio


def worker(task_queue, result_queue, args, id):
    device = torch.device(f"cuda:{args.gpu_id[id % args.gpu_num]}") if torch.cuda.is_available() else torch.device("cpu")
    while True:
        try:
            index, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        success, anomaly, total_distance, intensity, total_rotation_angle, arcs, arcs_num, mask_area_ratio = possess_single_row(row, index, args, device)
        result_queue.put((index, (success, anomaly, total_distance, intensity, total_rotation_angle, arcs, arcs_num, mask_area_ratio)))
        task_queue.task_done()


def parse_args():
    parser = argparse.ArgumentParser(description="Camera Pose Evaluation")
    parser.add_argument("csv_path", type=str, help="Path to the csv file")
    parser.add_argument("--dir_path", type=str, default="./outputs")
    parser.add_argument("--output_path", type=str, default="./outputs/evaluation_results.csv")
    parser.add_argument("--anomaly_threshold", type=int,
                        default=2, help="Threshold for anomaly detection")
    parser.add_argument('--gpu_id', type=str, default='0', help='Comma-separated list of GPU IDs to use')
    parser.add_argument("--num_workers", type=int,
                        default=4, help="Number of workers for parallel processing")
    parser.add_argument("--disable_parallel", action="store_true",
                        help="Disable parallel processing")
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_args()
    
    args.gpu_id = [int(gpu) for gpu in args.gpu_id.split(',')]
    args.gpu_num = len(args.gpu_id)

    df = pd.read_csv(args.csv_path)

    results = []
    if args.disable_parallel:
        for index, row in tqdm(df.iterrows(), total=len(df)):
            device = torch.device(f"cuda:{args.gpu_id[0]}") if torch.cuda.is_available() else torch.device("cpu")
            success, anomaly, total_distance, intensity, total_rotation_angle, arcs, arcs_num, mask_area_ratio = possess_single_row(row, index, args, device)
            results.append(index, (success, anomaly, total_distance, intensity, total_rotation_angle, arcs, arcs_num, mask_area_ratio))
    else:
        manager = Manager()
        task_queue = manager.Queue()
        for index, row in df.iterrows():
            task_queue.put((index, row))
            
        result_queue = manager.Queue()
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for id in range(args.num_workers):
                futures.append(executor.submit(worker, task_queue, result_queue, args, id))
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Finished workers"):
                future.result()
                
        # Collect results
        while not result_queue.empty():
            index, (success, anomaly, total_distance, intensity, total_rotation_angle, arcs, arcs_num, mask_area_ratio) = result_queue.get()
            results.append((index, (success, anomaly, total_distance, intensity, total_rotation_angle, arcs, arcs_num, mask_area_ratio)))

    results.sort(key=lambda x: x[0])
    # 拼接到原csv文件
    df['success'] = [result[1][0] for result in results]
    df['anomaly'] = [result[1][1] for result in results]
    df['total_distance'] = [result[1][2] for result in results]
    df['motion_intensity'] = [result[1][3] for result in results]
    df['camera_orient_angle'] = [result[1][4] for result in results]
    df['arcs'] = [result[1][5] for result in results]
    df['arcs_num'] = [result[1][6] for result in results]
    df['mask_area_ratio'] = [result[1][7] for result in results]

    # 保存结果
    df.to_csv(args.output_path, index=False)
    print(f"Results saved to {args.output_path}")
