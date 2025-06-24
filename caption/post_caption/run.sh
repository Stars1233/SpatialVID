#!/bin/bash

for i in {16..25}; do
    group_id="group_${i}"
    # 输出当前执行的group_id
    echo "当前: $group_id"
    python ./combine.py \
       --group_id $group_id
done