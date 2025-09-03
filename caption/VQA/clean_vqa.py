import os
import chardet
import multiprocessing
from tqdm import tqdm
import queue

def is_text_file(file_path):
    """
    Check if a file is a text file (by encoding detection and decoding test).
    Returns True if the file is likely a text file, False otherwise.
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(1024) 
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        if encoding is None:
            return False
            
        try:
            raw_data.decode(encoding)
            return True
        except UnicodeDecodeError:
            return False
    except Exception:
        return False

def contains_Scene_Description(file_path, encoding):
    """
    Check if the file contains the string 'Scene Description'.
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
            return 'Scene Description' in content
    except:
        return False

def worker(task_queue, result_queue, pbar, lock):
    """
    Worker process function: get files from the task queue and process them.
    Remove empty, broken, or invalid files and update progress.
    """
    while True:
        try:
            # Get task in non-blocking mode
            directory, filename = task_queue.get(block=False)
        except queue.Empty:
            # Exit loop if queue is empty
            break
            
        file_path = os.path.join(directory, filename)
        deleted = False
        
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            with lock:
                print(f"Delete empty file: {filename}")
            os.remove(file_path)
            deleted = True
        # Check if file is a valid text file
        elif not is_text_file(file_path):
            with lock:
                print(f"Delete broken file: {filename}")
                # Write error info and reason to error.log
                with open("vqa_error.log", "a") as f:
                    f.write(f"{filename} broken\n")
            os.remove(file_path)
            deleted = True
        # Check file encoding and content
        else:
            with open(file_path, 'rb') as f:
                raw_data = f.read(1024)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'  # Default to utf-8
            # Check if file contains "Scene Description"
            if not contains_Scene_Description(file_path, encoding):
                with lock:
                    print(f"Delete file without 'Scene Description': {filename}")
                    # Write error info and reason to error.log
                    with open("vqa_error.log", "a") as f:
                        f.write(f"{filename} does not contain 'Scene Description'\n")
                os.remove(file_path)
                deleted = True
        
        result_queue.put(deleted)
        with lock:
            pbar[0] += 1
        task_queue.task_done()

def clean_text_files(directory):
    """
    Clean broken or empty .txt files in the given directory.
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return
    # Get all .txt files
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    total_files = len(txt_files)
    if total_files == 0:
        print("No txt files found")
        return
    # Create queues
    task_queue = multiprocessing.JoinableQueue()
    result_queue = multiprocessing.Queue()
    lock = multiprocessing.Lock()
    # Fill task queue
    for filename in txt_files:
        task_queue.put((directory, filename))
    # Create shared progress counter
    manager = multiprocessing.Manager()
    pbar = manager.list([0])
    # Create and start worker processes
    processes = []
    for _ in range(32):
        p = multiprocessing.Process(
            target=worker,
            args=(task_queue, result_queue, pbar, lock)
        )
        p.start()
        processes.append(p)
    # Progress bar display process
    with tqdm(total=total_files, desc="VQA clean", unit="it") as pbar_display:
        last_count = 0
        while any(p.is_alive() for p in processes):
            current_count = pbar[0]
            if current_count > last_count:
                pbar_display.update(current_count - last_count)
                last_count = current_count
            import time
            time.sleep(0.1)
        task_queue.join()
        while True:
            try:
                pbar_display.update(result_queue.get_nowait())
            except queue.Empty:
                break
    # Wait for all processes to finish
    for p in processes:
        p.join()
    # Count deleted files
    deleted_count = 0
    while not result_queue.empty():
        if result_queue.get():
            deleted_count += 1
    print(f"total delete {deleted_count} files")

if __name__ == "__main__":
    import sys
    group_id = sys.argv[1]
    directory = f'/share/wjh/opencam/datasets/annotations/tmp_captions/{group_id}/vqa_tmp'
    clean_text_files(directory)
