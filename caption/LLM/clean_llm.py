import os
import chardet
import multiprocessing
from tqdm import tqdm
import queue
import time


def is_text_file(file_path):
    """
    Check if a file is a text file or contains non-text content
    Args:
        file_path: Path to the file to check
    Returns:
        bool: True if the file is a valid text file, False otherwise
    """
    try:
        # Read first 1024 bytes for encoding detection
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            
        # Detect file encoding
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        
        # If encoding can't be detected, it's likely not a text file
        if encoding is None:
            return False
            
        # Try decoding with detected encoding to verify
        try:
            raw_data.decode(encoding)
            return True
        except UnicodeDecodeError:
            return False
    except Exception:
        return False


# def contains_qwen_model(file_path, encoding):
#     """Check if file contains the string 'Qwen model'"""
#     try:
#         with open(file_path, 'r', encoding=encoding) as f:
#             content = f.read()
#             return 'Qwen model' in content
#     except:
#         return False


def worker(task_queue, result_queue, progress_counter, lock):
    """
    Worker process function that retrieves files from task queue and processes them
    Args:
        task_queue: Queue containing (directory, filename) tuples to process
        result_queue: Queue to store processing results (whether file was deleted)
        progress_counter: Shared counter to track processing progress
        lock: Multiprocessing lock for synchronized access to shared resources
    """
    while True:
        try:
            # Get task in non-blocking mode
            directory, filename = task_queue.get(block=False)
        except queue.Empty:
            # Exit loop when queue is empty
            break
            
        file_path = os.path.join(directory, filename)
        deleted = False
        
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            with lock:
                print(f"Deleting empty file: {filename}")
            os.remove(file_path)
            deleted = True
            
        # Check if file is a valid text file
        elif not is_text_file(file_path):
            with lock:
                print(f"Deleting corrupted file: {filename}")
            os.remove(file_path)
            deleted = True
            
        # Get file encoding and check content (commented out functionality)
        # else:
        #     with open(file_path, 'rb') as f:
        #         raw_data = f.read(1024)
        #         result = chardet.detect(raw_data)
        #         encoding = result['encoding'] or 'utf-8'  # Use utf-8 as fallback
                
        #     # Check if file contains "Qwen model"
        #     if not contains_qwen_model(file_path, encoding):
        #         with lock:
        #             print(f"Deleting file without 'Qwen model': {filename}")
        #         os.remove(file_path)
        #         deleted = True
        
        # Put result in queue
        result_queue.put(deleted)
        
        # Update shared progress counter
        with lock:
            progress_counter[0] += 1
        
        # Mark task as completed
        task_queue.task_done()


def clean_text_files(directory):
    """
    Clean up corrupted or empty text files in a directory using multiprocessing
    Args:
        directory: Path to the directory containing text files to clean
    """
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return
        
    # Get all txt files in the directory
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    total_files = len(txt_files)
    
    if total_files == 0:
        print("No text files found")
        return
    
    # Create task queue for file processing
    task_queue = multiprocessing.JoinableQueue()
    
    # Create result queue for tracking deletions
    result_queue = multiprocessing.Queue()
    
    # Create lock for synchronized printing and counter updates
    lock = multiprocessing.Lock()
    
    # Populate task queue with file paths
    for filename in txt_files:
        task_queue.put((directory, filename))
    
    # Create shared progress counter using manager
    manager = multiprocessing.Manager()
    progress_counter = manager.list([0])  # Use list to allow modification in processes
    
    # Create and start worker processes
    processes = []
    for _ in range(16):  # Use 16 worker processes
        p = multiprocessing.Process(
            target=worker, 
            args=(task_queue, result_queue, progress_counter, lock)
        )
        p.start()
        processes.append(p)
    
    # Display progress bar
    with tqdm(total=total_files, desc="Cleaning files", unit="file") as pbar_display:
        last_count = 0
        # Update progress bar while processes are running
        while any(p.is_alive() for p in processes):
            current_count = progress_counter[0]
            if current_count > last_count:
                pbar_display.update(current_count - last_count)
                last_count = current_count
            time.sleep(0.1)  # Short sleep to reduce CPU usage
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Count total deleted files
    deleted_count = 0
    while not result_queue.empty():
        if result_queue.get():
            deleted_count += 1
    
    print(f"Total deleted files: {deleted_count}")


if __name__ == "__main__":
    import sys
    # Get group ID from command line argument
    if len(sys.argv) < 2:
        print("Usage: python clean_text_files.py <group_id>")
        sys.exit(1)
        
    group_id = sys.argv[1]
    directory = f'/share/wjh/opencam/datasets/annotations/tmp_captions/{group_id}/llm_tmp'
    clean_text_files(directory)
    