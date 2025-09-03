import numpy as np
import os
import requests


def npy_to_jsonl(npy_path, output_dir):
    """
    Convert npy file to jsonl format, keeping only the first row for every 5 rows
    Data processing: Multiply each value by 1000000, round to integer, and keep only first 3 columns
    Args:
        npy_path: Path to the npy file
        output_dir: Directory for the output file
    """
    # Ensure the output directory exists, create if it doesn't
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data from npy file
    data = np.load(npy_path)
    
    # Generate output file path
    output_filename = os.path.join(output_dir, 'poses_selected.jsonl')
    
    # Write data to jsonl file, taking only the first row for every 5 rows
    with open(output_filename, 'w') as f:
        # Iterate through data with step size 5
        for i in range(0, len(data), 5):
            if i < len(data):  # Ensure index doesn't exceed array bounds
                row = data[i]  # Take first row of each 5-row group
                
                # Process data: keep first 3 columns, multiply by 1e6, round to integer
                processed_row = np.round(row[:3] * 1000000).astype(int).tolist()
                print(processed_row)
                
                # Convert to string format and write to file
                txt = ' '.join([str(x) for x in processed_row]) + '\n'
                print(txt)
                f.write(txt)
    
    print(f"File saved to: {output_filename}")


def get_pose(group_id, scene_id):
    """
    Retrieve and process pose data from extrinsics.npy file
    Args:
        group_id: Group identifier (e.g., 'group_0')
        scene_id: Scene identifier (e.g., '0ewQTRGsenQ_157')
    Returns:
        Formatted string of pose data, or None if error occurs
    """
    # Base directory for pose data
    data_dir = "/share/wjh/opencam/datasets/annotations/cameras"
    pose_path = os.path.join(data_dir, group_id, scene_id, 'extrinsics.npy')
    
    # Check if pose file exists
    if not os.path.exists(pose_path):
        print(f"Pose file {pose_path} does not exist.")
        
        # Log error if not already logged
        with open('llm_error.log', 'r') as f:
            if f"{group_id}/{scene_id}:Pose file does not exist.\n" not in f.readlines():
                with open('llm_error.log', 'a') as f:
                    f.write(f"{group_id}/{scene_id}:Pose file does not exist.\n")
        return None
    
    # Try to load and process the pose file
    try:
        poses = np.load(pose_path)
    except Exception as e:
        print(f"Failed to load pose file {pose_path}: {str(e)}")
        
        # Log error if not already logged
        with open('llm_error.log', 'r') as f:
            error_msg = f"{group_id}/{scene_id}:Failed to load pose file {pose_path}: {str(e)}\n"
            if error_msg not in f.readlines():
                with open('llm_error.log', 'a') as f:
                    f.write(error_msg)
        return None
    
    # Data processing steps
    poses = poses[::5, :, 3]  # Take first row for every 5 rows
    max_value = np.max(poses)
    min_value = np.min(poses)
    min_abs_value = np.min(np.abs(poses))
    
    # Normalize and convert to integers (minimize integer digits)
    poses = np.round(poses / (max_value - min_value) / min_abs_value).astype(int)
    
    # Keep only first 3 columns and transpose
    poses = poses[:, :3].T
    
    # Extract individual axes
    poses1, poses2, poses3 = poses[0], poses[1], poses[2]
    
    # Convert each axis to string
    poses1_str = ' '.join(map(str, poses1))
    poses2_str = ' '.join(map(str, poses2))
    poses3_str = ' '.join(map(str, poses3))
    
    # Combine into formatted string
    poses_str = f'x:{poses1_str}\ny:{poses2_str}\nz:{poses3_str}'
    
    return poses_str


def get_prompt(group_id, scene_id, prompt_dir, vqa_caption, motion_intensity):
    """
    Construct a prompt by combining content from p1.txt, p2.txt, VQA caption, and pose data
    Args:
        group_id: Group identifier
        scene_id: Scene identifier
        prompt_dir: Directory containing p1.txt and p2.txt
        vqa_caption: VQA result caption
        motion_intensity: Motion intensity information
    Returns:
        Combined prompt string, or None if pose data is missing
    """
    # Read prompt components
    p1_file = os.path.join(prompt_dir, 'p1.txt')
    p2_file = os.path.join(prompt_dir, 'p2.txt')
    
    with open(p1_file, 'r', encoding='utf-8') as f:
        p1_content = f.read().strip()
    
    with open(p2_file, 'r', encoding='utf-8') as f:
        p2_content = f.read().strip()
    
    # Get pose data
    poses = get_pose(group_id, scene_id)
    if poses is None:
        return None
    
    # Assemble final prompt
    prompt = (f"{p1_content}\nGiven Information:\n{vqa_caption}\n3.Camera Position Data:\n{poses}\n"
              f"\n4.Motion intensity:\n{motion_intensity}\n{p2_content}")
    
    return prompt


def api_call(group_id, scene_id, prompt_dir, model, api_key, vqa_caption, motion_intensity, base_domain):
    """
    Call API with constructed prompt and return the response
    Args:
        prompt_dir: Directory containing p1.txt and p2.txt
        group_id: Group identifier
        scene_id: Scene identifier
        model: Name of the model to use
        api_key: API key (optional, uses default if not provided)
        vqa_caption: VQA result caption
        motion_intensity: Motion intensity information
        base_domain: Base domain for the API
    Returns:
        API response content as string, or None if error occurs
    """
    # Get the constructed prompt
    prompt_text = get_prompt(group_id, scene_id, prompt_dir, vqa_caption, motion_intensity)
    if prompt_text is None:
        print(f"Prompt for {group_id}/{scene_id} could not be created.")
        return None
    
    # Determine API configuration
    is_qwen = "dashscope.aliyuncs.com" in base_domain
    api_url = f"{base_domain}v1/chat/completions"
    
    # Prepare payload based on API type
    if is_qwen:
        # Configuration for Qwen model API
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "enable_thinking": False,
            "temperature": 0.1
        }
    else:
        # Configuration for other APIs
        messages_content = [{"type": "text", "text": prompt_text}]
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": messages_content}
            ],
            "temperature": 0.1,
            "user": "DMXAPI"
        }
    
    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    # Add Qwen-specific headers if needed
    if not is_qwen:
        headers["User-Agent"] = f"DMXAPI/1.0.0 ({base_domain})"
    
    # Make API request
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()  # Raise exception for HTTP errors
        response_data = response.json()
        
        # Extract response content based on API type
        if is_qwen:
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    except Exception as e:
        print(f"API request error: {str(e)}")
        return None
