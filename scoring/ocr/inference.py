import os
import glob
import argparse
import pandas as pd
from multiprocessing import Manager
import queue
import concurrent.futures
from tqdm import tqdm
import cv2
import glob

from paddleocr import PaddleOCR

def process_single_row(row, args, model):
    img_dir = os.path.join(args.fig_load_dir, row["id"])
    img_list = sorted(glob(f"{img_dir}/img/*.jpg"))[:3]
    # 读取图片
    images = [cv2.imread(img_path) for img_path in img_list]
    
    result = model.predict(input=images)
    area = row['height'] * row['width']
    
    area_list = []
    for res in result:
        total_text_area = 0  # 初始化总文本区域面积
        for rec_box in res["rec_boxes"]:
            x_min, y_min, x_max, y_max = float(rec_box[0]), float(rec_box[1]), float(rec_box[2]), float(rec_box[3])  # 输出左上角和右下角坐标
            text_area = (x_max - x_min) * (y_max - y_min)  # 计算文本区域面积
            total_text_area += text_area
        ratio = total_text_area / area
        area_list.append(ratio)
    return max(area_list) if area_list else 0.0  # 返回最大面积比率，如果没有检测到文本区域则返回0.0

def worker(task_queue, result_queue, args, id):
    gpu_id = id % args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # 绑定GPU
    device = "gpu:0" #if torch.cuda.is_available() else "cpu"

    model = PaddleOCR(
        device=device,
        use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
        use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
        use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
    )

    while True:
        try:
            index, row = task_queue.get_nowait()
        except queue.Empty:
            break
        
        area_list = process_single_row(row, args, model)
        result_queue.put((index, area_list))

def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 Image Predictor")
    parser.add_argument('csv_path', type=str, help='Path to the csv file')
    parser.add_argument("--fig_load_dir", type=str, default="img", help="Directory containing images")
    parser.add_argument("--num_workers", type=int, default=16, help="#workers for concurrent.futures")
    parser.add_argument("--gpu_num", type=int, default=1, help="gpu number")
    parser.add_argument("--skip_if_existing", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.csv_path):
        print(f"Meta file '{args.csv_path}' not found. Exit.")
        return

    wo_ext, ext = os.path.splitext(args.csv_path)
    out_path = f"{wo_ext}_ocr{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    df = pd.read_csv(args.csv_path)

    manager = Manager()
    task_queue = manager.Queue()
    result_queue = manager.Queue()
    for index, row in df.iterrows():
        task_queue.put((index, row))

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for id in range(args.num_workers):
            futures.append(executor.submit(worker, task_queue, result_queue, args, id))
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Finished workers"):
            future.result()

    results = []
    while not result_queue.empty():
        index, area_list = result_queue.get()
        results.append((index, area_list))
    results.sort(key=lambda x: x[0])
    df["ocr"] =  [x[1] for x in results]

    df.to_csv(out_path, index=False)
    print(f"New meta (shape={df.shape}) with ocr results saved to '{out_path}'.")

if __name__ == "__main__":
    main()
