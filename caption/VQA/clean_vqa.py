import os
import chardet
import multiprocessing
from tqdm import tqdm
import queue

def is_text_file(file_path):
    """检查文件是否为文本文件或包含非文本内容"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(1024)  # 读取前1024字节用于检测
            
        # 检测文件编码
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        # 如果无法检测编码，可能不是文本文件
        if encoding is None:
            return False
            
        # 尝试使用检测到的编码解码
        try:
            raw_data.decode(encoding)
            return True
        except UnicodeDecodeError:
            return False
    except Exception:
        return False

def contains_Scene_Description(file_path, encoding):
    """检查文件是否包含'Scene Description'字符串"""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
            return 'Scene Description' in content
    except:
        return False

def worker(task_queue, result_queue, pbar, lock):
    """工作进程函数，从任务队列获取文件并处理"""
    while True:
        try:
            # 非阻塞方式获取任务
            directory, filename = task_queue.get(block=False)
        except queue.Empty:
            # 队列为空时退出循环
            break
            
        file_path = os.path.join(directory, filename)
        deleted = False
        
        # 检查文件是否为空
        if os.path.getsize(file_path) == 0:
            with lock:
                print(f"删除空文件: {filename}")
            os.remove(file_path)
            deleted = True
            
        # 检查文件是否为文本文件
        elif not is_text_file(file_path):
            with lock:
                print(f"删除破损文件: {filename}")
            os.remove(file_path)
            deleted = True
            
        # 获取文件编码并检查内容
        else:
            with open(file_path, 'rb') as f:
                raw_data = f.read(1024)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'  # 默认使用utf-8
                
            # 检查文件是否包含"Scene Description"
            if not contains_Scene_Description(file_path, encoding):
                with lock:
                    print(f"删除不包含'Scene Description'的文件: {filename}")
                os.remove(file_path)
                deleted = True
        
        # 将结果放入结果队列
        result_queue.put(deleted)
        
        # 更新共享计数器而不是直接更新进度条
        with lock:
            pbar[0] += 1
        
        # 标记任务完成
        task_queue.task_done()

def clean_text_files(directory):
    """清理目录中的破损或空txt文件"""
    if not os.path.exists(directory):
        print(f"错误: 目录 '{directory}' 不存在")
        return
        
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    total_files = len(txt_files)
    
    if total_files == 0:
        print("没有找到txt文件")
        return
    
    # 创建任务队列
    task_queue = multiprocessing.JoinableQueue()
    
    # 创建结果队列
    result_queue = multiprocessing.Queue()
    
    # 创建进程间锁，用于控制打印输出
    lock = multiprocessing.Lock()
    
    # 填充任务队列
    for filename in txt_files:
        task_queue.put((directory, filename))
    
    # 创建共享的进度计数器
    manager = multiprocessing.Manager()
    pbar = manager.list([0])  # 用列表模拟共享变量
    
    # 创建并启动工作进程
    processes = []
    for _ in range(32):
        p = multiprocessing.Process(
            target=worker, 
            args=(task_queue, result_queue, pbar, lock)
        )
        p.start()
        processes.append(p)
    
    # 创建进度条显示进程
    with tqdm(total=total_files, desc="VQA clean", unit="it") as pbar_display:
        # 监控共享计数器并更新进度条
        last_count = 0
        while any(p.is_alive() for p in processes):
            current_count = pbar[0]
            if current_count > last_count:
                pbar_display.update(current_count - last_count)
                last_count = current_count
            # 短暂休眠以减少CPU使用
            import time
            time.sleep(0.1)
        # 等待所有任务完成
        task_queue.join()
        
        # 更新进度条
        while True:
            try:
                pbar_display.update(result_queue.get_nowait())
            except queue.Empty:
                break
    
    # 等待所有进程结束
    for p in processes:
        p.join()
    
    # 统计删除的文件数
    deleted_count = 0
    while not result_queue.empty():
        if result_queue.get():
            deleted_count += 1
    
    print(f"total delete {deleted_count} files")

if __name__ == "__main__":
    # 指定要清理的目录路径
    # 从argv获取group id，只接受一个参数
    import sys
    group_id = sys.argv[1]
    directory = f'/share/wjh/opencam/datasets/annotations/tmp_captions/{group_id}/vqa_tmp'
    clean_text_files(directory)
