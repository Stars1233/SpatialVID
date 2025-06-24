import numpy as np
import json
import os
import requests

def npy_to_jsonl(npy_path, output_dir):
    """
    将npy文件转换为jsonl格式，每5行数据只保存第一行
    数据处理：将每个数值乘以1000000并四舍五入保留整数，只保留前三列
    Args:
        npy_path: npy文件路径
        output_dir: 输出目录
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取npy文件
    data = np.load(npy_path)
    
    # 生成输出文件名
    output_filename = os.path.join(output_dir, 'poses_selected.jsonl')
    
    # 将数据写入jsonl文件，每5行只取第一行
    with open(output_filename, 'w') as f:
        for i in range(0, len(data), 5):
            if i < len(data):  # 确保索引不超出范围
                row = data[i]  # 每5行取第一行
                # 只保留前三列，将每个数值乘以1000000并四舍五入保留整数
                processed_row = np.round(row[:3] * 1000000).astype(int).tolist()
                print(processed_row)
                txt = ' '.join([str(x) for x in list(processed_row)])+'\n'
                print(txt)
                f.write(txt)
    
    print(f"文件已保存至: {output_filename}")

def get_pose(group_id, scene_id):
    # [group_id, scene_id] for example: group_id='group_0', scene_id='0ewQTRGsenQ_157'
    data_dir="/share/wjh/opencam/datasets/annotations/cameras"
    pose_path = os.path.join(data_dir, group_id, scene_id, 'extrinsics.npy')
    if not os.path.exists(pose_path):
        print(f"Pose file {pose_path} does not exist.")
        return None
        # raise FileNotFoundError(f"Pose file {pose_path} does not exist.")
    poses = np.load(pose_path)
    poses = poses[::5, :, 3] # 每5行取第一行
    max_value = np.max(poses)
    min_value = np.min(poses)
    min_abs_value = np.min(np.abs(poses))
    # 将poses归一化并除以最小值，再转为整数（最小化整数位数，并且随着poses的shape变化而变化）
    poses = np.round(poses / (max_value - min_value) / min_abs_value).astype(int) # shape, (poses.shape[0], 3)
    # 提取前三列
    poses = poses[:, :3]
    # 转置
    poses = poses.T
    # 分别提取三行
    poses1 = poses[0]
    poses2 = poses[1]
    poses3 = poses[2]
    # 分别转成3个字符串
    poses1_str = ' '.join(map(str, poses1))
    poses2_str = ' '.join(map(str, poses2))
    poses3_str = ' '.join(map(str, poses3))
    # 拼接成一个字符串
    poses_str = 'x:' + poses1_str + '\ny:' + poses2_str + '\nz:' + poses3_str
    
    return poses_str


def get_prompt(group_id, scene_id, prompt_dir, vqa_caption):
    """
    从prompt_dir目录读取p1.txt和p2.txt文件
    将三个部分按顺序拼接成最终的提示词（p1+vqa_caption+poses信息+p2.txt）
    """
    # 读取p1.txt和p2.txt文件
    p1_file = os.path.join(prompt_dir, 'p1.txt')
    p2_file = os.path.join(prompt_dir, 'p2.txt')
    with open(p1_file, 'r', encoding='utf-8') as f:
        p1_content = f.read().strip()
    with open(p2_file, 'r', encoding='utf-8') as f:
        p2_content = f.read().strip()
    # 读取poses.jsonl文件
    poses = get_pose(group_id, scene_id)
    if poses is None:
        # print(f"Pose file for {group_id}/{scene_id} does not exist.")
        return None
    # 拼接最终的提示词
    prompt = p1_content +"\nGiven Information:\n"+ vqa_caption + "\n3.Camera Position Data:\n" + poses +"\n"+ p2_content
    # prompt = p1_content + vqa_caption + "\n" + p2_content
    # print(prompt)
    return prompt
    
def api_call(group_id, scene_id, prompt_dir, model, api_key, vqa_caption, Base_Domain):
    """
    输入prompt，调用API并保存结果（文件名前加final，保存到other_data目录）
    Args:
        prompt_dir: 包含p1.txt和p2.txt的目录路径
        group_id: 批次序号
        scene_id: clip的ID
        model: 使用的模型名称
        api_key: API密钥（可选，未提供则用默认）
        vqa_caption: VQA结果
        Base_Domain: API基础域名
    Returns:
        str: API响应内容
    """

    prompt_text = get_prompt(group_id, scene_id, prompt_dir, vqa_caption)
    if prompt_text is None:
        print(f"Prompt file for {group_id}/{scene_id} does not exist.")
        return None
    
    # 构建请求内容
    BASE_DOMAIN = Base_Domain
    
    # 判断是否为千问模型API
    is_qwen = "dashscope.aliyuncs.com" in BASE_DOMAIN
    
    if is_qwen:
        API_URL = BASE_DOMAIN + "v1/chat/completions"
        # 千问模型的消息格式
        payload = {
            "model": model,
            "messages": [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            "enable_thinking": False,
            "temperature": 0.1
        }
    else:
        # 原有API的格式
        API_URL = BASE_DOMAIN + "v1/chat/completions"
        messages_content = [
            {"type": "text", "text": prompt_text}
        ]
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": messages_content}
            ],
            "temperature": 0.1,
            "user": "DMXAPI"
        }

    # 设置请求头
    if is_qwen:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
    else:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": f"DMXAPI/1.0.0 ({BASE_DOMAIN})",
            "Accept": "application/json"
        }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        response_data = response.json()

        # 打印输入和输出的token数量
        # if 'usage' in response_data:
        #     usage = response_data.get('usage', {})
        #     prompt_tokens = usage.get('prompt_tokens', 0)
        #     completion_tokens = usage.get('completion_tokens', 0)
        #     total_tokens = usage.get('total_tokens', 0)
            
        #     print(f"输入tokens: {prompt_tokens}")
        #     print(f"输出tokens: {completion_tokens}")
        #     print(f"总tokens: {total_tokens}")
        # else:
        #     print("API响应中未包含token使用信息")

        # 根据不同API格式获取响应内容
        if is_qwen:
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

    except Exception as e:
        print(f"API请求错误：{str(e)}")
        return None
