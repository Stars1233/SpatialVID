# video caption

## Environment

```
requests
Pillow
numpy
tqdm
argparse
pandas
```

## VQA

### 获取API中转商的密钥和基础域名（两个供应商选其一即可）

- https://next.ohmygpt.com/apis/keys （比较稳定）
    1.  （可选）任选其一，修改为/VQA/run.sh中的`base_domain`
        
        ```
        https://www.aigptx.top
        https://cn2us02.opapi.win
        ```
        
    2. （必选）获取密钥，并对/VQA/run.sh中的`api_key`进行修改
- https://www.dmxapi.cn/ （便宜，但不那么稳定）
    1. 修改`base_domain`为
        
        ```
        https://www.dmxapi.cn
        ```
        
    2. 获取密钥，并对/VQA/run.sh中的`api_key`进行修改

<aside>

⚠️所选base_domain和api_key必须属于同一厂商！

</aside>

### 快速开始

1. 将/VQA/run.sh中的`group_id`修改为数据批次
2. `num_workers`代表线程数，`wait_time`代表每发送一个线程的等待时间（太小会报错）
3. 接下来可以开始运行
    
    ```bash
    cd VQA
    ./run.sh
    ```
    

### ⚠️注意事项

- 请确保csv文件的第2列是id
- 请确保clip图片位置和命名正确：
代码会默认寻找每个clip下的img子目录，并抽取其中的jpg文件（每5张抽1张）
例如：clip目录为`data_test/stage1_total_done_sample_200/*b*-rKdc58w_54`，代码会寻找`data_test/stage1_total_done_sample_200/*b*-rKdc58w_54/img/[test].jpg`

## LLM

### 调用LLM

1. 获取Qwen密钥
官方平台链接为：
https://bailian.console.aliyun.com/?tab=model#/model-market
将密钥加入`/LLM/api_list.txt`
    
2. 相应地修改/LLM/run.sh中的`group_id`
3. 接下来可以开始运行

    ```bash
    cd LLM
    ./run.sh