import os
import sys
import csv
import json
import re
import argparse
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import concurrent.futures
from multiprocessing import Manager
import queue

# 从split.py导入parse_text_to_json函数
def parse_text_to_json(text: str) -> Dict[str, Any]:
    """
    将特定格式的文本解析为JSON结构
    
    Args:
        text: 输入的特定格式文本
    
    Returns:
        解析后的JSON字典
    """
    # 定义标签及其在JSON中的对应键名
    labels = {
        "Camera Motion Caption": "OptCamMotion",
        "Scene Abstract Caption": "SceneSummary",
        "Main Motion Trend Summary": "MotionTrends",
        "Scene Keywords": "SceneTags",
        "Immersive Shot Summary": "ShotImmersion",
        "Qwen model": "LLM"
    }
    
    result = {key: "" for key in labels.values()}
    current_label = None
    current_content = []
    
    # 按行处理文本
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 检查是否为标签行
        for label, json_key in labels.items():
            if label in line:
                # 找到标签后的第一个字母位置
                start_pos = line.find(label) + len(label)
                # 跳过非字母字符
                while start_pos < len(line) and not line[start_pos].isalpha():
                    start_pos += 1
                
                content = line[start_pos:].strip()
                
                # 如果标签后有内容，直接处理
                if content:
                    if json_key in ["MotionTrends", "SceneTags"]:
                        # 只按逗号（中英文）分割，保留空格作为短语的一部分
                        items = re.split(r'[，,]\s*', content)
                        result[json_key] = [item.strip() for item in items if item.strip()]
                    else:
                        result[json_key] = content
                    
                    current_label = None
                    current_content = []
                else:
                    # 标签后无内容，继续读取后续行
                    current_label = json_key
                    current_content = []
                
                break
        else:
            # 如果当前有正在收集的标签内容
            if current_label:
                # 检查行是否为空
                if line:
                    # 找到行的第一个字母
                    start_pos = 0
                    while start_pos < len(line) and not line[start_pos].isalpha():
                        start_pos += 1
                    
                    if start_pos < len(line):
                        current_content.append(line[start_pos:])
                else:
                    # 遇到空行，结束当前标签内容的收集
                    content = ' '.join(current_content).strip()
                    
                    if current_label in ["MotionTrends", "SceneTags"]:
                        # 只按逗号（中英文）分割，保留空格作为短语的一部分
                        items = re.split(r'[，,]\s*', content)
                        result[current_label] = [item.strip() for item in items if item.strip()]
                    else:
                        result[current_label] = content
                    
                    current_label = None
                    current_content = []
        
        i += 1
    
    # 处理Qwen model标签，它可能延伸到最后
    if current_label == "LLM" and current_content:
        content = ' '.join(current_content).strip()
        result[current_label] = content
    
    return result

def vqa_parse_text_to_json(text: str) -> Dict[str, Any]:
    """
    将包含Camera Motion Caption和Scene Description的文本解析为JSON格式
    
    Args:
        text: 输入的特定格式文本
    
    Returns:
        解析后的JSON字典
    """
    result = {
        "CamMotion": "",
        "SceneDesc": ""
    }
    
    # 处理Camera Motion Caption - 从标签后第一个字母开始到换行
    camera_pattern = r'Camera Motion Caption:\s*(\w[\s\S]*?)(?=\n|$)'
    camera_match = re.search(camera_pattern, text)
    if camera_match:
        result["CamMotion"] = camera_match.group(1).strip()
    
    # 处理Scene Description - 从标签后第一个字母开始到文本末尾
    scene_pattern = r'Scene Description:\s*(\w[\s\S]*)$'
    scene_match = re.search(scene_pattern, text, re.DOTALL)
    if scene_match:
        result["SceneDesc"] = scene_match.group(1).strip()
    
    return result

def process_single_row(group_id: str, scene_id: str, num_workers: int = 4) -> None:
    """
    处理单个场景的VQA和LLM字幕，合并为一个JSON
    
    Args:
        group_id: 批次ID
        scene_id: 场景ID
        num_workers: 工作线程数量
    """
    # 定义路径
    tmp_path = "/share/wjh/opencam/datasets/annotations/tmp_captions"
    vqa_path = os.path.join(tmp_path, f"{group_id}/vqa_tmp")
    llm_path = os.path.join(tmp_path, f"{group_id}/llm_tmp")
    output_path = f"/share/wjh/opencam/datasets/annotations/captions/{group_id}"
    
    # 确保输出目录存在
    # os.makedirs(output_path, exist_ok=True)
    
    # 构建文件路径
    vqa_file = os.path.join(vqa_path, f"{scene_id}.txt")
    llm_file = os.path.join(llm_path, f"{scene_id}.txt")
    output_file = os.path.join(output_path, f"{scene_id}.json")
    
    # 如果输出文件已存在，则跳过
    if os.path.exists(output_file):
        # print(f"文件已存在，跳过：{output_file}")
        return
    
    # 检查输入文件是否存在
    if not os.path.isfile(vqa_file):
        error_log_path = "combine_error.log"
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{group_id}/{scene_id}\n")
        print(f"未找到VQA文件: {vqa_file}")
        return
    
    if not os.path.isfile(llm_file):
        error_log_path = "combine_error.log"
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{group_id}/{scene_id}\n")
        print(f"未找到LLM文件: {llm_file}")
        return
    
    try:
        # 读取VQA文件
        with open(vqa_file, 'r', encoding='utf-8') as f:
            vqa_text = f.read()
        
        # 读取LLM文件
        with open(llm_file, 'r', encoding='utf-8') as f:
            llm_text = f.read()
        
        # 解析为JSON
        vqa_json = vqa_parse_text_to_json(vqa_text)
        llm_json = parse_text_to_json(llm_text)
        
        # 合并JSON
        combined_json = {**vqa_json, **llm_json}
        
        # 保存到输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(combined_json, f, ensure_ascii=False, indent=2)
        
        # print(f"JSON数据已保存到 {output_file}")
    
    except Exception as e:
        error_log_path = "combine_error.log"
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{group_id}/{scene_id}\n")
        print(f"处理{scene_id}时发生错误：{e}")

def process(group_id: str, num_workers: int = 4) -> None:
    """
    处理指定批次的所有场景
    
    Args:
        group_id: 批次ID
        num_workers: 工作线程数量
    """
    # 定义路径
    base_path = "/share/wjh/opencam/meta_infos/sam2_infos"
    csv_path = os.path.join(base_path, f"{group_id}.csv")
    
    # 检查CSV文件是否存在
    if not os.path.isfile(csv_path):
        print(f"未找到CSV文件: {csv_path}")
        return

    output_path = f"/share/wjh/opencam/datasets/annotations/captions/{group_id}"
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 读取CSV文件
    rows = []
    try:
        with open(csv_path, "r", encoding="utf-8") as fin:
            reader = csv.reader(fin)
            header = next(reader)  # 跳过表头
            rows = list(reader)
    except Exception as e:
        print(f"读取CSV文件时发生错误：{e}")
        return
    
    # 代表scene_id的列索引
    subdir_col_idx = 1
    
    # 使用多线程处理
    manager = Manager()
    task_queue = manager.Queue()
    
    # 将任务加入队列
    for idx, row in enumerate(rows):
        scene_id = row[subdir_col_idx]
        task_queue.put((idx, scene_id))
    
    # 定义工作线程函数
    def worker(task_queue, pbar):
        while True:
            try:
                idx, scene_id = task_queue.get(timeout=1)
            except queue.Empty:
                break
            
            process_single_row(group_id, scene_id, num_workers)
            task_queue.task_done()
            pbar.update(1)
    
    # 启动多线程处理
    with tqdm(total=len(rows), desc="处理进度") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                futures.append(executor.submit(worker, task_queue, pbar))
            for future in concurrent.futures.as_completed(futures):
                future.result()
    
    print(f"批次 {group_id} 处理完成，共处理 {len(rows)} 个场景")

def parse_args():
    parser = argparse.ArgumentParser(description='合并VQA和LLM字幕数据')
    parser.add_argument('--group_id', type=str, required=True, help='批次ID')
    parser.add_argument('--num_workers', type=int, default=4, help='工作线程数量')
    return parser.parse_args()

def main() -> None:
    """
    主函数，处理命令行参数
    """
    args = parse_args()
    process(args.group_id, args.num_workers)

if __name__ == "__main__":
    main()