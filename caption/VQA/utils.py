import base64
import requests
import sys
import os
from PIL import Image  # 新增
import cv2
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def encode_image(image):
    """
    读取本地图片，压缩到640x360，并编码为Base64字符串（带data URI前缀）
    """
    # 打开图片并压缩
    # with Image.open(image_path) as img:
    #     img = img.convert("RGB")
    #     img = img.resize((640, 360))
    #     # img = img.resize((512, 288))
    #     # 保存到内存
    #     from io import BytesIO
    #     buffer = BytesIO()
    #     img.save(buffer, format="JPEG")
    #     buffer.seek(0)
    #     base64_data = base64.b64encode(buffer.read()).decode("utf-8")
    #     ext = "jpeg"
    #     return f"data:image/{ext};base64,{base64_data}"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 360))
    _, buffer = cv2.imencode('.jpeg', image)
    base64_data = base64.b64encode(buffer).decode("utf-8")
    ext = "jpeg"
    return f"data:image/{ext};base64,{base64_data}"

def api_call(video_path, video_fps, video_num_frames, prompt_text, model, api_key, Base_Domain):
    """
    处理指定目录下的图片并调用API
    Args:
        video_path: 视频文件路径
        video_fps: 视频帧率
        video_num_frames: 视频总帧数
        prompt_text: 提示词文本
        model: 使用的模型名称
        api_key: API密钥（可选，未提供则用默认）
    Returns:
        dict: API响应结果
    """
    # 检查video path
    if not os.path.exists(video_path):
        print(f"视频文件不存在：{video_path}")
        return None

    # Extract image
    images = []
    cap = cv2.VideoCapture(video_path)
    interval = int(video_fps * 0.2) * 5
    for frame in range(video_num_frames):
        ret, image = cap.read()
        if not ret:
            break
        if frame % interval == 0:
            images.append(image)

    # 直接使用传入的prompt_text
    if not prompt_text:
        print("未提供提示词")
        return None

    # 构建多模态输入内容，先初始化为空列表
    messages_content = []
    
    # 先处理多张图片
    for image in images:
        try:
            encoded_image = encode_image(image)
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": encoded_image}
            })
        except Exception as e:
            print(f"图片处理错误：{str(e)}")
            return None
    # 再加入prompt
    messages_content.append({"type": "text", "text": prompt_text})

    # opmygpt
    BASE_DOMAIN = Base_Domain
    API_URL = BASE_DOMAIN + "v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": messages_content}
        ],
        "temperature": 0.1,
        "user": "DMXAPI"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": f"DMXAPI/1.0.0 ({BASE_DOMAIN})",
        "Accept": "application/json"
    }

    try:
        # print("API请求中...")  # 注释掉
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=300
        )
        response.raise_for_status()
        response_data = response.json()
        
        return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
    except Exception as e:
        print(f"API请求错误：{str(e)}")
        return None
