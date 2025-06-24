import os
import pandas as pd
import numpy as np
import sys
from utils import npy_to_jsonl

def process_csv_for_npy(data_dir, csv_filename):
    """
    遍历指定CSV文件中的名称，寻找相应子目录并调用npy_to_jsonl处理数据
    
    Args:
        data_dir: 数据根目录
        csv_filename: CSV文件名（包含要处理的子目录名列表）
    """
    # 读取CSV文件
    csv_path = os.path.join(data_dir, csv_filename)
    if not os.path.exists(csv_path):
        print(f"错误：CSV文件不存在 - {csv_path}")
        return
    
    try:
        # 读取CSV文件，假设第一列包含子目录名
        df = pd.read_csv(csv_path)
        # 获取第一列的列名
        first_column = df.columns[1]
        # 获取子目录名列表
        subdirs = df[first_column].tolist()
        
        # 处理每个子目录
        for subdir in subdirs:
            # 构建子目录完整路径
            subdir_path = os.path.join(data_dir, subdir)
            
            if not os.path.exists(subdir_path):
                print(f"警告：子目录不存在 - {subdir_path}")
                continue
            
            # 查找子目录下的reconstructions/poses.npy文件
            npy_path = os.path.join(subdir_path, 'reconstructions', 'poses.npy')
            
            if not os.path.exists(npy_path):
                print(f"警告：在子目录 {subdir_path} 中未找到 reconstructions/poses.npy 文件")
                continue
            
            # 默认输出到子目录下的other_data目录
            target_output_dir = os.path.join(subdir_path, 'other_data')
            
            # 确保输出目录存在
            if not os.path.exists(target_output_dir):
                os.makedirs(target_output_dir)
            
            # 调用npy_to_jsonl函数处理数据
            print(f"处理文件: {npy_path}")
            print(f"输出目录: {target_output_dir}")
            npy_to_jsonl(npy_path, target_output_dir)
        
        print("处理完成！")
    
    except Exception as e:
        print(f"处理CSV文件时出错: {str(e)}")

if __name__ == "__main__":
    # 处理命令行参数
    if len(sys.argv) < 3:
        print("用法: python process_npy.py <数据目录> <CSV文件名>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    csv_filename = sys.argv[2]
    
    # 调用处理函数
    process_csv_for_npy(data_dir, csv_filename)