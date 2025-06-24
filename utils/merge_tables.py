import os
import glob
import argparse
import pandas as pd


def read_csv_file(file_path):
    """
    读取单个 CSV 文件
    """
    return pd.read_csv(file_path)


def merge_tables_from_files(file_list, output_file, merge_on=None):
    """
    Merge multiple CSV files. By default, it is assumed that all columns except the last one in each file
    are the common columns used as merge keys, and the last column contains different data.
    The original column names are preserved.

    Parameters:
      file_list: list, containing the paths of CSV files.
      output_file: str, the path for the merged CSV file.
      merge_on: list or None, list of column names used for merging. If None, automatically selects all columns except the last one in each file.
    """
    if not file_list:
        raise ValueError("File list is empty!")

    # 读取所有 CSV 文件
    dfs = [read_csv_file(f) for f in file_list]

    # 自动选择合并键：前 13 列
    if merge_on is None:
        merge_on = dfs[0].columns[:13].tolist()

    # 合并数据帧
    df_merged = dfs[0]
    for df in dfs[1:]:
        # 检查合并键是否一致
        if merge_on != df.columns[:13].tolist():
            raise ValueError(f"The common columns in one of the files are not consistent with previous files!")
        # 根据 merge_on 键进行合并；每个文件的其余列保留其原始名称
        df_merged = pd.merge(df_merged, df, on=merge_on)

    # 保存合并结果
    df_merged.to_csv(output_file, index=False)
    print(f"Merge completed. The merged file is saved as {output_file}")
    return df_merged


def main():
    parser = argparse.ArgumentParser(description="Merge multiple CSV files from a specified folder")
    parser.add_argument("folder", type=str, help="Path to the folder containing CSV files")
    parser.add_argument("--output", type=str, required=True, help="Path for the merged CSV file")

    args = parser.parse_args()

    # 仅匹配前缀为 'clips_info_' 的 CSV 文件
    pattern = os.path.join(args.folder, "clips_info_*.csv")
    file_list = glob.glob(pattern)
    file_list.sort()  # 排序以确保合并顺序一致
    if not file_list:
        raise ValueError(f"No CSV files matching the condition were found in folder {args.folder}!")

    print(f"Found {len(file_list)} CSV files:")
    for f in file_list:
        print(f"  {f}")

    # 调用合并函数
    merge_tables_from_files(file_list, args.output)


if __name__ == "__main__":
    main()