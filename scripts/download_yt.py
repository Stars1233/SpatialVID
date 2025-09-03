'''
Author: jeremiah.wang
Date: 2024-11-22 16:12:02
LastEditTime: 2025-02-25 22:17:28
Description: yt-dlp download video from youtube
'''
# from https://huggingface.co/Ligeng-Zhu/panda70m-download
import sys, os, os.path as osp
import yt_dlp
import asyncio
from concurrent.futures import ProcessPoolExecutor
import fire
import pandas as pd
import json
import time

def ytb_download(url, json_info, output_dir="ytb_videos/"):
    """
    Download a specified YouTube video using yt-dlp and save related metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    uid = url.split("?v=")[-1]
    yt_opts = {
    # "format": "bv[height=720][ext=mp4]"
    # "format": "bv[height=720]",  # Download the best quality available
    "format": "bv", # for panda70m
    # "format": "bv[height=720][ext=mp4][vcodec!^=av]", 
    "proxy": "127.0.0.1:7893",
    "outtmpl": osp.join(output_dir, f"{uid}.%(ext)s"),  # Output template
    # "cookiesfrombrowser": "chrome",  # Use Chrome's cookies automatically
    # "cookiefile": "cookies.txt",  # Use a custom cookies file
    # "postprocessors": [
    #     {
    #         "key": "FFmpegVideoConvertor",
    #         "preferedformat": "mp4",  # Convert video to mp4 format (slow)
    #     }
    # ],
    # "verbose" : True,
    'abort-on-error': True, # Abort downloading when an error occurs
    'retries': 60, # Number of retries
    'ffmpeg_location': '/usr/bin/ffmpeg', # Path to ffmpeg
    'quiet': True, # Suppress output
    'sleep-requested': 5, # Sleep for 1.25 seconds between requests
    'min-sleep-interval': 60,
    'max-sleep-interval': 90,
    }
    video_path_mp4 = osp.join(output_dir, f"{uid}.mp4")
    video_path_webm = osp.join(output_dir, f"{uid}.webm")
    meta_path = osp.join(output_dir, f"{uid}.json")
    if (osp.exists(video_path_mp4) or osp.exists(video_path_webm)) and osp.exists(meta_path):
        print(f"\033[91m{uid} already labeled.\033[0m")
        return 0
    try:
        with yt_dlp.YoutubeDL(yt_opts) as ydl:
            ydl.download([url])
        with open(meta_path, "w") as fp:
            json.dump(json_info, fp, indent=2)
        return 0
    except Exception as e:
        print(f"\033[91mError downloading {url}: {e}\033[0m")
        err_map = {
            'Requested format is not available': "z0322_dld_format_noavailable.log",
            'removed by': "z0322_dld_removed_by.log",
            'Private video': "z0322_dld_private_video.log",
        }
        for key, log_file in err_map.items():
            if key in str(e):
                with open(osp.join(output_dir, f"{log_file}"), "a") as f:
                    f.write(f"{url}\n")
                break
        else:
            with open(osp.join(output_dir, f"z0322_dld_othererr.log"), "a") as f:
                f.write(f"{url}, {str(e)}\n")
        return -1




async def main(csv_path, max_workers=10, shards=0, total=-1, limit=False):
    """
    Batch download YouTube videos specified in a CSV file, supporting sharding and concurrency.
    """
    base_dir = '/path/to/raw_videos'
    PPE = ProcessPoolExecutor(max_workers=max_workers)
    loop = asyncio.get_event_loop()
    df = pd.read_csv(csv_path)
    csv_path = os.path.basename(csv_path)
    output_dir = f'{base_dir}/{csv_path.split(".")[0]}'
    data_list = list(df.iterrows())
    if total > 0:
        chunk = len(data_list) // total
        begin_idx = shards * chunk
        end_idx = (shards + 1) * chunk if shards < total - 1 else len(data_list)
        data_list = data_list[begin_idx:end_idx]
    print(f"download total {len(data_list)} videos")
    tasks = []
    for idx, (index, row) in enumerate(data_list):
        video_url = row["url"]
        json_info = {"caption": row["caption"]}
        tasks.append(loop.run_in_executor(PPE, ytb_download, video_url, json_info, output_dir))
        if limit and idx >= 20:
            break
    res = await asyncio.gather(*tasks)
    print(f"[{sum(res)} / {len(res)}]")




def entry(csv="meta_data_sample_500.csv", shards=0, total=-1, limit=False, max_workers=2):
    """
    Command line entry function, supports fire invocation.
    """
    print(csv, shards, total, max_workers)
    start_time = time.time()
    print(f"\033[92mStarting execution at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\033[0m")
    asyncio.run(main(csv, max_workers=max_workers, shards=shards, total=total, limit=limit))
    end_time = time.time()
    print(f"\033[92mFinished execution at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\033[0m")
    print(f"\033[92mTotal execution time: {end_time - start_time:.2f} seconds\033[0m")


        
def add_download(csv_path):
    """
    Download missing videos according to the new_vid_path field in the CSV file.
    """
    data = pd.read_csv(csv_path)
    for _, row in data.iterrows():
        video_path = row['new_vid_path']
        video_url = os.path.splitext(os.path.basename(video_path))[0]
        video_url = f'https://www.youtube.com/watch?v={video_url}'
        ytb_download(video_url, json_info={}, output_dir='videos/')
        print(f"Downloaded {video_url} to {video_path}")
if __name__ == "__main__":
    # Call entry function via command line arguments
    fire.Fire(entry)
    # To supplement download: add_download(csv_path='xxx.csv')
