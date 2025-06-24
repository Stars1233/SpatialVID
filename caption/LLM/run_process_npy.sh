#!/bin/bash

# 脚本名称: run_process_npy.sh
# 描述: 调用process_npy.py处理CSV和NPY文件
# 用法: 直接运行此脚本，参数已在脚本中显式指定

# 显式指定参数
DATA_DIR="/home/zrj/project/api_test/pipeline/data"
CSV_FILENAME="stage1_total_done_sample_200.csv"

# 检查数据目录是否存在
if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据目录 '$DATA_DIR' 不存在"
    exit 1
fi

# 检查CSV文件是否存在
CSV_PATH="$DATA_DIR/$CSV_FILENAME"
if [ ! -f "$CSV_PATH" ]; then
    echo "错误: CSV文件 '$CSV_PATH' 不存在"
    exit 1
fi

# 设置Python脚本路径
SCRIPT_PATH="./process_npy.py"

# 检查Python脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: Python脚本 '$SCRIPT_PATH' 不存在"
    exit 1
fi

# 根据是否提供输出目录来调用Python脚本

echo "执行: python $SCRIPT_PATH 处理 $DATA_DIR 中的 $CSV_FILENAME"
python "$SCRIPT_PATH" "$DATA_DIR" "$CSV_FILENAME"

# 检查Python脚本执行状态
if [ $? -eq 0 ]; then
    echo "处理完成！"
else
    echo "处理过程中出现错误，请检查日志。"
    exit 1
fi