#!/bin/bash

for i in {0..0}; do
    group_id="group_${i}"
    echo "Now: $group_id"
    python ./combine.py \
       --group_id $group_id
done