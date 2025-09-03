import sys
import os
import base64
import json
import requests
import cv2
from PIL import Image

# Add parent directory to system path for module imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def encode_image(image):
    """
    Resizes an image to 640x360 and encodes it as a Base64 string with data URI prefix.
    
    Args:
        image: OpenCV image object (BGR format)
        
    Returns:
        str: Base64 encoded image string with data URI prefix
    """
    # Resize image to standard dimensions (640x360)
    resized_image = cv2.resize(image, (640, 360))
    
    # Encode image as JPEG and convert to Base64
    _, buffer = cv2.imencode('.jpeg', resized_image)
    base64_data = base64.b64encode(buffer).decode("utf-8")
    
    # Return with data URI format for API compatibility
    return f"data:image/jpeg;base64,{base64_data}"

def api_call(video_path, video_fps, video_num_frames, prompt_text, model, api_key, base_domain):
    """
    Extracts key frames from a video, constructs a multimodal request, and calls the API.
    
    Args:
        video_path (str): Path to the video file
        video_fps (float): Frames per second of the video
        video_num_frames (int): Total number of frames in the video
        prompt_text (str): Text prompt for the API request
        model (str): Name of the model to use for the API call
        api_key (str): API authentication key
        base_domain (str): Base domain URL for the API endpoint
        
    Returns:
        str: API response content if successful, None otherwise
    """
    # Validate video file existence
    if not os.path.exists(video_path):
        print(f"Video file does not exist: {video_path}")
        return None

    # Extract key frames from video (one frame per second)
    # Calculation: 0.2 seconds * 5 = 1 second interval between frames
    frames = []
    video_capture = cv2.VideoCapture(video_path)
    frame_interval = int(video_fps * 0.2 * 5)  # Interval between frames in number of frames
    
    for frame_idx in range(video_num_frames):
        ret, frame = video_capture.read()
        if not ret:  # End of video or read error
            break
        if frame_idx % frame_interval == 0:  # Capture frame at specified interval
            frames.append(frame)

    # Validate prompt text
    if not prompt_text:
        print("No prompt text provided")
        return None

    # Construct multimodal input content
    messages_content = []
    
    # Add encoded images to request content
    for frame in frames:
        try:
            encoded_frame = encode_image(frame)
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": encoded_frame}
            })
        except Exception as e:
            print(f"Image processing error: {str(e)}")
            return None
    
    # Add text prompt to request content
    messages_content.append({"type": "text", "text": prompt_text})

    # Configure API request parameters
    api_url = f"{base_domain}v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": messages_content}],
        "temperature": 0.1,  # Low temperature for more deterministic responses
        "user": "opencam"
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    # Execute API request with error handling
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=300  # 5-minute timeout for long processing
        )
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Extract and return the response content
        response_data = response.json()
        return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
    except Exception as e:
        try:
            # Attempt to parse JSON error response
            error_data = response.json()
            print(f"API request error: {str(e)}")
            print(f'Total frames processed: {len(frames)}, Video FPS: {video_fps}')
            print(f"Error details: {json.dumps(error_data, indent=2)}")
        except Exception as inner_e:
            # Handle non-JSON error responses
            print(f"Exception type: {type(inner_e)}, Error message: {str(inner_e)}")
            print(f"API request error: {str(e)}")
            print(f"Response content: {response.text}")
        return None
