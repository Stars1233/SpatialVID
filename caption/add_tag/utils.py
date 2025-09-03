import numpy as np
import os
import json
import requests


def get_description(save_file):
    """
    Retrieve the description from the 'SceneDesc' key in a JSON file.
    The JSON file is typically located at: 
    /share/wjh/opencam/datasets/annotations/captions/[group_id]/[scene_id].json
    
    Args:
        save_file (str): Path to the JSON file containing the scene description
        
    Returns:
        str: The scene description from 'SceneDesc' key, or None if file not found
    """
    try:
        # Open and load the JSON file
        with open(save_file, 'r') as f:
            data = json.load(f)
            description = data['SceneDesc']
            return description
    except FileNotFoundError:
        print(f"File not found: {save_file}")
        return None


def get_prompt(save_file, prompt_file):
    """
    Construct a complete prompt by combining content from a prompt file 
    with a scene description extracted from a JSON file.
    
    The final prompt is formed by concatenating:
    [Content from prompt_file] + [Scene description from save_file]
    
    Args:
        save_file (str): Path to JSON file with scene description
        prompt_file (str): Path to text file containing base prompt content (p1)
        
    Returns:
        str: Constructed prompt string, or None if description retrieval fails
    """
    # Read base prompt content from prompt file
    with open(prompt_file, 'r', encoding='utf-8') as f:
        p1_content = f.read().strip()
    
    # Get scene description
    description = get_description(save_file)
    if description is None:
        return None
    
    # Combine base prompt and description
    prompt = p1_content + description
    # print(prompt)  # Optional: Uncomment to verify prompt construction
    return prompt


def api_call(save_file, prompt_file, model, api_key, base_domain):
    """
    Make an API call to a language model with a constructed prompt,
    handling different API formats for different model providers.
    
    Args:
        save_file (str): Path to JSON file with scene description
        prompt_file (str): Path to text file with base prompt content
        model (str): Name of the model to use for the API call
        api_key (str): API key for authentication
        base_domain (str): Base domain URL for the API endpoint
        
    Returns:
        str: Response content from the API, or None if request fails
    """
    # Generate the complete prompt
    prompt_text = get_prompt(save_file, prompt_file)
    if prompt_text is None:
        print(f"Prompt file for {save_file} does not exist.")
        return None
    
    # Determine if using Qwen model API (Aliyun)
    is_qwen = "dashscope.aliyuncs.com" in base_domain
    
    # Configure API endpoint and payload based on model type
    if is_qwen:
        api_url = base_domain + "v1/chat/completions"
        # Payload format specific to Qwen model
        payload = {
            "model": model,
            "messages": [
                # {"role": "system", "content": "You are a helpful assistant."},  # Optional system message
                {"role": "user", "content": prompt_text}
            ],
            "enable_thinking": False,
            "temperature": 0.1  # Low temperature for more deterministic output
        }
    else:
        # Payload format for other models
        api_url = base_domain + "v1/chat/completions"
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

    # Configure request headers
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
            "User-Agent": f"DMXAPI/1.0.0 ({base_domain})",
            "Accept": "application/json"
        }

    try:
        # Execute API request with timeout
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=120  # 2-minute timeout
        )
        response.raise_for_status()  # Raise exception for HTTP errors
        response_data = response.json()

        # Optional: Uncomment to log token usage
        # if 'usage' in response_data:
        #     usage = response_data.get('usage', {})
        #     prompt_tokens = usage.get('prompt_tokens', 0)
        #     completion_tokens = usage.get('completion_tokens', 0)
        #     total_tokens = usage.get('total_tokens', 0)
        #     
        #     print(f"Input tokens: {prompt_tokens}")
        #     print(f"Output tokens: {completion_tokens}")
        #     print(f"Total tokens: {total_tokens}")
        # else:
        #     print("API response does not contain token usage information")

        # Extract and return response content based on API format
        if is_qwen:
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

    except Exception as e:
        print(f"API request error: {str(e)}")
        return None
