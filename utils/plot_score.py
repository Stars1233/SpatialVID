import argparse
import os
import pandas as pd
import concurrent.futures
import queue
from random import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def row_test(x):
    # if x.name % 1000 == 0:'
    print(len(x))
    if x.name == 1000:
        print(x.name)
        print(x)
    return True


def col_test(x):
    print(x)
    return True


def plot_keywords(meta, keywords, bins=10, base_name='', save_dir=''):
    """
    Plot histograms for the specified keywords in the meta DataFrame.

    Args:
        meta (pd.DataFrame): The DataFrame containing the metadata.
        keywords (list): List of keywords to plot.
        bins (int): Number of bins for the histogram.
    """
    GRID = False
    sparse_keys = []
    for keyword in keywords:
        if keyword in meta.columns:
            plt.figure(figsize=(10, 6))
            print('-' * 100)
            if type(meta[keyword][0]) == str or meta[keyword] in sparse_keys:
                print(f"Skipping {keyword} because it is a string or already in sparse_keys")
                # 如果为离散值，绘制条形图
                counts = meta[keyword].value_counts()
                plt.bar(counts.index, counts.values, label=keyword, alpha=0.7, edgecolor='black')

                # Set x-axis ticks to display every third tick for 'resolution_text'
                if keyword == 'resolution_text':
                    x_ticks = counts.index[::3]
                    plt.xticks(x_ticks, rotation=45)
            else:
                data = meta[keyword].dropna()
                # 使用数据本身的最小值和最大值
                x_min, x_max = data.min(), data.max()
                plt.hist(data, bins=bins, label=keyword, alpha=0.7, edgecolor='black')

                # 根据不同的关键字设置不同的间隔
                if keyword in ['flow_mean', 'flow_max', 'flow_min', 'lum_mean', 'lum_max', 'lum_min', 'motion']:
                    x_min = 0  # 固定从0开始
                    interval = (x_max - x_min) / bins * 2
                else:
                    interval = (x_max - x_min) / bins * 3

                # 设置x轴刻度
                plt.xticks(np.arange(x_min, x_max + interval, interval))

            plt.xlabel(keyword)
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {keyword}')
            plt.legend()
            plt.grid(GRID)

            path = f"{save_dir}/{base_name}-{'grid-' if GRID else ''}{bins}-{keyword}-histogram.png"
            plt.savefig(path)
            print(f"Histogram of {keyword} saved as [{path}]")
        else:
            print(f"Column '{keyword}' not found in the meta DataFrame.")


def tex_res(height, width):
    return f"{height}x{width}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str)
    parser.add_argument("--fig_save_dir", type=str, required=True, help="Directory to save the figure")
    parser.add_argument("--num_workers", type=int, default=None, help="#workers for concurrent.futures")
    parser.add_argument("--disable-parallel", action="store_true", help="disable parallel processing")
    args = parser.parse_args()
    return args


def process_row(row):
    return tex_res(row['height'], row['width'])


def process_fps(row):
    return str(int(row['fps']))


def worker(task_queue, results_queue):
    while True:
        try:
            index, task_type, row = task_queue.get(timeout=1)
        except queue.Empty:
            break
        if task_type == 'resolution_text':
            result = process_row(row)
        elif task_type == 'fps':
            result = process_fps(row)
        results_queue.put((index, task_type, result))
        task_queue.task_done()


def main():
    args = parse_args()
    meta_path = args.meta_path

    wo_ext, ext = os.path.splitext(meta_path)
    tag = wo_ext.split('/')[-2]
    base_name = f"{tag}"
    # check if the meta file exists
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        return

    # load the meta file
    meta = pd.read_csv(meta_path)

    if args.disable_parallel:
        meta['resolution_text'] = [process_row(row) for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing resolution_text")]
        meta['fps'] = [process_fps(row) for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Processing fps")]
    else:
        from multiprocessing import Manager
        manager = Manager()
        task_queue = manager.Queue()
        results_queue = manager.Queue()

        for index, row in meta.iterrows():
            task_queue.put((index, 'resolution_text', row))
            task_queue.put((index, 'fps', row))

        if args.num_workers is not None:
            num_workers = args.num_workers
        else:
            num_workers = os.cpu_count()

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for _ in range(num_workers):
                future = executor.submit(worker, task_queue, results_queue)
                futures.append(future)

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Finished workers"):
                future.result()

        res_text_results = [None] * len(meta)
        fps_results = [None] * len(meta)
        while not results_queue.empty():
            index, task_type, result = results_queue.get()
            if task_type == 'resolution_text':
                res_text_results[index] = result
            elif task_type == 'fps':
                fps_results[index] = result

        meta['resolution_text'] = res_text_results
        meta['fps'] = fps_results

    # analyze height and width
    heights = meta['height'].value_counts()
    widths = meta['width'].value_counts()
    fps = meta['fps'].value_counts()
    # print(f"Heights: {heights}")
    # print(f"Widths: {widths}")
    print(f"FPS: {fps}")
    # print(list(meta['id']))
    keywords = [
        'num_frames',
        'aes',
        'flow_mean',
        'flow_max',
        'flow_min',
        'cmotion',
        'ocr',
        'lum_mean',
        'lum_max',
        'lum_min',
        'blur',
        'resolution_text',
        # 'resolution',
        'fps',
        'motion',
    ]
    bins = [
        40,
        # 80,
        # 120
    ]  # Set the number of bins for the histogram
    if len(bins) == 1:
        bins = bins * len(keywords)
    print(base_name)
    for i, keyword in enumerate(keywords):
        plot_keywords(meta, [keyword], bins=bins[i], base_name=base_name, save_dir=args.fig_save_dir)


if __name__ == "__main__":
    main()