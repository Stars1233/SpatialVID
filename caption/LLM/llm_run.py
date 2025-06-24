import os
import sys
import csv
import concurrent.futures
from multiprocessing import Manager
import queue
from tokenize import group
from tqdm import tqdm
import argparse
import random
import time

sys.path.append(os.path.dirname(__file__))
from utils import api_call

def process_single_row(group_id, scene_id, prompt_dir, model, api_key, vqa_caption, Base_Domain="https://cn2us02.opapi.win/"):
    tmp_path = "/share/wjh/opencam/datasets/annotations/tmp_captions"
    if not os.path.isdir(tmp_path):
        print(f"未找到子目录: {tmp_path}")
        return
    LLM_path = os.path.join(tmp_path, f"{group_id}/llm_tmp")
    if not os.path.isdir(LLM_path):
        os.makedirs(LLM_path, exist_ok=True)
        print(f"已创建{LLM_path}")
    
    save_file = os.path.join(LLM_path,f"{scene_id}.txt")
    # 如果文件已存在，则跳过
    if os.path.exists(save_file):
        # print(f"文件已存在，跳过：{save_file}")
        return

    llm_caption = api_call(group_id, scene_id, prompt_dir, model, api_key, vqa_caption, Base_Domain)
    cnt = 15
    while llm_caption is None and cnt > 0:
        llm_caption = api_call(group_id, scene_id, prompt_dir, model, api_key, vqa_caption, Base_Domain)
        time.sleep(2)
        cnt -= 1
    if llm_caption is None:
        # 将错误的{group_id}/{scene_id}写入到当前目录的error.log里
        error_log_path = "llm_error.log"
        with open(error_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{group_id}/{scene_id}\n")
        print(f"API请求错误或未获取到结果：{scene_id}")
        return

    # 将llm_caption结果加上“\n 5. Qwen model: [model_name]”写入save_file
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write(llm_caption + f"\n\n6. Qwen model: \n{model}")
    return

def process(group_id, prompt_dir, model_file, api_key_file, num_workers=4, wait_time=0.8, Base_Domain="https://cn2us02.opapi.win/", record_time=False):
    import time
    
    # 读取模型和API密钥文件
    with open(model_file, "r") as mf:
        models = [line.strip() for line in mf.readlines()]
    with open(api_key_file, "r") as af:
        api_keys = [line.strip() for line in af.readlines()]

    tmp_path = "/share/wjh/opencam/datasets/annotations/tmp_captions"
    if not os.path.isdir(tmp_path):
        print(f"未找到子目录: {tmp_path}")
        
        
    LLM_path = os.path.join(tmp_path, f"{group_id}/llm_tmp")
    if not os.path.isdir(LLM_path):
        os.makedirs(LLM_path, exist_ok=True)
        print(f"已创建{LLM_path}")

    vqa_path = os.path.join(tmp_path, f"{group_id}", "vqa_tmp")
    if not os.path.isdir(vqa_path):
        print(f"未找到子目录: {vqa_path}")
        return

    base_path = "/share/wjh/opencam/meta_infos/sam2_infos"
    csv_path = os.path.join(base_path,f"{group_id}.csv")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        rows = list(reader)

    # 代表idx的列
    subdir_col_idx = 1 
    manager = Manager()
    total_time = 0

    def worker(task_queue, pbar):
        import time
        while True:
            try:
                idx_row = task_queue.get(timeout=1)
            except queue.Empty:
                break
            time.sleep(wait_time)

            idx, row = idx_row
            scene_id = row[subdir_col_idx]

            vqa_txt_path = os.path.join(vqa_path, f"{scene_id}.txt")
            if not os.path.isfile(vqa_txt_path):
                error_log_path = "llm_error.log"
                with open(error_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{group_id}/{scene_id}\n")
                print(f"未找到文件: {vqa_txt_path}")
                continue
            with open(vqa_txt_path, "r") as f:
                vqa_caption = f.read()
            
            # 加上随机性，使每个模型和密钥使用次数更均衡
            model_num = len(models)
            key_num = len(api_keys)
            model_idx = random.randint(0, model_num - 1)
            key_idx = random.randint(0, key_num - 1)
            model = models[model_idx]
            api_key = api_keys[key_idx]
            # print(f"当前使用的模型: {model}")
            # print(f"当前使用的API序号: {key_idx}")
            process_single_row(group_id, scene_id, prompt_dir, model, api_key, vqa_caption, Base_Domain)
            task_queue.task_done()
            pbar.update(1)
            

    manager = Manager()
    task_queue = manager.Queue()
    for idx, row in enumerate(rows):
        task_queue.put((idx, row))

    start_time = time.time() if record_time else None
    with tqdm(total=len(rows), desc=f"({group_id})LLM Finished") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                futures.append(executor.submit(worker, task_queue, pbar))
            for future in concurrent.futures.as_completed(futures):
                future.result()
    
    if record_time:
        total_time = time.time() - start_time
    
    if record_time and len(rows) > 0:
        print(f"API总用时: {total_time:.2f}秒, 平均每个子目录用时: {total_time/len(rows):.2f}秒")
    elif record_time:
        print("没有有效的子目录，无法统计用时。")

# 修改parse_args函数以接收新的参数

def parse_args():
    parser = argparse.ArgumentParser(description='VQA处理程序')
    parser.add_argument('--group_id', type=str, help='批次序号')
    parser.add_argument('--prompt_dir', type=str, 
                        default=os.path.join(os.path.dirname(__file__), "vqa_prompt.txt"),
                        help='提示词文件路径')
    parser.add_argument('--model_file', type=str, default="model_list.txt",
                        help='模型文件路径')
    parser.add_argument('--api_key_file', type=str, default="api_list.txt",
                        help='API密钥文件路径')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='工作线程数量')
    parser.add_argument('--wait_time', type=float, default=0.8,
                        help='每个线程每次请求间隔的时间（秒）')
    parser.add_argument('--base_domain', type=str, default="https://cn2us02.opapi.win/",
                        help='API域名')
    parser.add_argument('--record_time', action='store_true',
                        help='是否记录处理时间')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process(
        group_id=args.group_id,
        prompt_dir=args.prompt_dir,
        model_file=args.model_file,
        api_key_file=args.api_key_file,
        num_workers=args.num_workers,
        wait_time=args.wait_time,
        Base_Domain=args.base_domain,
        record_time=args.record_time
    )