import cv2
import os
import datetime
import subprocess
import json
from PIL import Image
import numpy as np

# https://github.com/lkk688/VisionLangAnnotate/blob/main/backend/src/extractframefromvideo.py
# video from https://www.pexels.com/video/cars-on-highway-854671/

def extract_metadata(video_path):
    """Extract metadata including GPS information from video file using ffprobe."""
    cmd = [
        'ffprobe', 
        '-v', 'quiet', 
        '-print_format', 'json', 
        '-show_format', 
        '-show_streams', 
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        metadata = json.loads(result.stdout)
        
        # Extract GPS data if available
        gps_info = {}
        creation_time = None
        
        # Look for creation time and GPS data in format tags
        if 'format' in metadata and 'tags' in metadata['format']:
            tags = metadata['format']['tags']
            
            # Extract creation time
            time_fields = ['creation_time', 'date', 'com.apple.quicktime.creationdate']
            for field in time_fields:
                if field in tags:
                    creation_time = tags[field]
                    break
            
            # Common GPS metadata fields
            gps_fields = [
                'location', 'location-eng', 'GPS', 
                'GPSLatitude', 'GPSLongitude', 'GPSAltitude',
                'com.apple.quicktime.location.ISO6709'
            ]
            
            for field in gps_fields:
                if field in tags:
                    gps_info[field] = tags[field]
        
        # Also check stream metadata for creation time if not found
        if creation_time is None and 'streams' in metadata:
            for stream in metadata['streams']:
                if 'tags' in stream and 'creation_time' in stream['tags']:
                    creation_time = stream['tags']['creation_time']
                    break
        
        # If no creation time found, use file modification time
        if creation_time is None:
            file_mtime = os.path.getmtime(video_path)
            creation_time = datetime.datetime.fromtimestamp(file_mtime).isoformat()
        
        return metadata, gps_info, creation_time
    
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return None, {}, None

def resize_with_aspect_ratio(image, target_size):
    """
    Resize image maintaining aspect ratio.
    
    Parameters:
    - image: PIL Image or numpy array
    - target_size: Tuple of (width, height) representing the maximum dimensions
    
    Returns:
    - Resized PIL Image
    """
    if isinstance(image, np.ndarray):
        # Convert OpenCV image (numpy array) to PIL
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Get original dimensions
    original_width, original_height = image.size
    target_width, target_height = target_size
    
    # Calculate aspect ratios
    original_aspect = original_width / original_height
    target_aspect = target_width / target_height
    
    # Determine new dimensions maintaining aspect ratio
    if original_aspect > target_aspect:
        # Width constrained
        new_width = target_width
        new_height = int(target_width / original_aspect)
    else:
        # Height constrained
        new_height = target_height
        new_width = int(target_height * original_aspect)
    
    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_image

def extract_key_frames(video_path, output_dir, target_size=(640, 480), extraction_method="scene_change", privacy_blur=False):
    """
    Extract key frames from a video file and save them with timestamp names.
    
    Parameters:
    - video_path: Path to the input video file
    - output_dir: Directory to save extracted frames
    - target_size: Tuple of (width, height) maximum dimensions for resizing
    - extraction_method: Method to extract frames ('scene_change', 'interval', or 'both')
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get video metadata
    metadata, gps_info, creation_time = extract_metadata(video_path)
    
    # Save metadata to a JSON file
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump({
            'video_metadata': metadata, 
            'gps_info': gps_info, 
            'creation_time': creation_time
        }, f, indent=4)
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate video creation time based on metadata or file timestamp
    video_creation_datetime = None
    if creation_time:
        try:
            # Try different time formats
            for time_format in [
                "%Y-%m-%dT%H:%M:%S.%fZ", 
                "%Y-%m-%dT%H:%M:%SZ", 
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"
            ]:
                try:
                    video_creation_datetime = datetime.datetime.strptime(creation_time, time_format)
                    break
                except ValueError:
                    continue
        except:
            pass  # Use None if parsing fails
    
    print(f"Video Information:")
    print(f"- Frame Rate: {fps} fps")
    print(f"- Frame Count: {frame_count}")
    print(f"- Resolution: {original_width}x{original_height}")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"- Creation Time: {creation_time}")
    print(f"- GPS Info: {gps_info}")
    
    # Initialize variables
    prev_frame = None
    frame_idx = 0
    saved_count = 0
    
    # Parameters for scene change detection
    min_scene_change_threshold = 30.0  # Minimum threshold for scene change
    frame_interval = int(fps) * 1  # Save a frame every second as fallback
    
    if privacy_blur ==True:
        # Load detection models
        print("Loading privacy models...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        face_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
        face_model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small").to(device)
        
        plate_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        plate_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection").to(device)
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale for scene change detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        should_save = False
        reason = ""
        
        # Method 1: Detect scene changes
        if extraction_method in ["scene_change", "both"]:
            if prev_frame is not None:
                # Calculate mean absolute difference between current and previous frame
                diff = cv2.absdiff(gray, prev_frame)
                mean_diff = np.mean(diff)
                
                if mean_diff > min_scene_change_threshold:
                    should_save = True
                    reason = f"scene_change (diff={mean_diff:.2f})"
        
        # Method 2: Save frames at regular intervals
        if extraction_method in ["interval", "both"]:
            if frame_idx % frame_interval == 0:
                should_save = True
                reason = "interval"
        
        # Save the frame if needed
        if should_save:
            # Calculate timestamp in the video
            timestamp_seconds = frame_idx / fps
            timestamp = str(datetime.timedelta(seconds=int(timestamp_seconds)))
            milliseconds = int((timestamp_seconds - int(timestamp_seconds)) * 1000)
            timestamp = f"{timestamp}.{milliseconds:03d}"
            
            # Calculate frame creation time if video creation time is available
            frame_creation_time = None
            if video_creation_datetime:
                frame_creation_time = (video_creation_datetime + 
                                      datetime.timedelta(seconds=timestamp_seconds)).isoformat()
            
            if privacy_blur == True:
                frame = perform_privacyblur(
                        frame, 
                        face_model=face_model, 
                        plate_model=plate_model,
                        device=device
                    )
                
            # Resize the frame maintaining aspect ratio
            pil_img = resize_with_aspect_ratio(frame, target_size)
            new_width, new_height = pil_img.size
            
            # Save the frame
            filename = f"frame_{timestamp.replace(':', '-')}_{reason}.jpg"
            output_path = os.path.join(output_dir, filename)
            pil_img.save(output_path, quality=95)
            
            # Save frame metadata
            frame_meta = {
                "frame_index": frame_idx,
                "timestamp_seconds": timestamp_seconds,
                "timestamp": timestamp,
                "extraction_reason": reason,
                "original_dimensions": {
                    "width": original_width,
                    "height": original_height
                },
                "resized_dimensions": {
                    "width": new_width,
                    "height": new_height
                },
                "video_creation_time": creation_time,
                "frame_creation_time": frame_creation_time,
                "extraction_time": datetime.datetime.now().isoformat()
            }
            
            # Add GPS data to frame metadata
            if gps_info:
                frame_meta["gps_info"] = gps_info
            
            # Save frame metadata
            frame_meta_file = os.path.join(output_dir, f"{filename.replace('.jpg', '.json')}")
            with open(frame_meta_file, 'w') as f:
                json.dump(frame_meta, f, indent=4)
            
            saved_count += 1
            print(f"Saved frame {saved_count}: {filename} ({reason})")
        
        # Update variables for next iteration
        prev_frame = gray.copy()
        frame_idx += 1
    
    # Release resources
    cap.release()
    print(f"Extraction complete. Saved {saved_count} key frames to {output_dir}")

def perform_privacyblur(frame, face_model=None, plate_model=None, device=None, confidence_threshold=0.5):
    """
    Detect and blur faces and license plates in a video frame for privacy compliance.
    
    Args:
        frame: numpy array image frame from video
        face_model: pre-loaded face detection model (if None, will load default)
        plate_model: pre-loaded license plate detection model (if None, will load default)
        device: torch device to use (if None, will use GPU if available)
        confidence_threshold: minimum confidence score for detections
    
    Returns:
        numpy array of the frame with faces and license plates blurred
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models if not provided
    if face_model is None:
        print("Loading face detection model...")
        face_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-small")
        face_model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small").to(device)
    
    if plate_model is None:
        print("Loading license plate detection model...")
        # Using a general object detection model that can detect license plates
        # You might want to fine-tune this for better license plate detection
        plate_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        plate_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection").to(device)
    
    # Make a copy of the original frame
    blurred_frame = frame.copy()
    
    # Detect faces
    face_inputs = face_processor(images=frame, return_tensors="pt").to(device)
    face_outputs = face_model(**face_inputs)
    face_results = face_processor.post_process_object_detection(face_outputs, threshold=confidence_threshold)[0]
    
    # Blur faces
    for score, label, box in zip(face_results["scores"], face_results["labels"], face_results["boxes"]):
        if face_model.config.id2label[label.item()] == "person":
            # Extract face coordinates and add some margin
            x1, y1, x2, y2 = box.int().tolist()
            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Extract the region and apply blur
            face_region = blurred_frame[y1:y2, x1:x2]
            blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
            blurred_frame[y1:y2, x1:x2] = blurred_face
    
    # Detect license plates
    plate_inputs = plate_processor(images=frame, return_tensors="pt").to(device)
    plate_outputs = plate_model(**plate_inputs)
    plate_results = plate_processor.post_process_object_detection(plate_outputs, threshold=confidence_threshold)[0]
    
    # Blur license plates
    for score, label, box in zip(plate_results["scores"], plate_results["labels"], plate_results["boxes"]):
        # For general object detection models, look for relevant categories
        label_name = plate_model.config.id2label[label.item()]
        if any(keyword in label_name.lower() for keyword in ["car", "vehicle", "truck", "plate"]):
            # Extract license plate coordinates
            x1, y1, x2, y2 = box.int().tolist()
            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            # Extract the region and apply blur
            plate_region = blurred_frame[y1:y2, x1:x2]
            blurred_plate = cv2.GaussianBlur(plate_region, (99, 99), 30)
            blurred_frame[y1:y2, x1:x2] = blurred_plate
    
    return blurred_frame

if __name__ == "__main__":
    video_path = "data/video/test_vid.mp4"
    output_dir = "data/images"

    extract_key_frames(
        video_path=video_path,
        output_dir=output_dir,
        target_size=(640, 640),
        extraction_method="interval",
        privacy_blur=False
    )
