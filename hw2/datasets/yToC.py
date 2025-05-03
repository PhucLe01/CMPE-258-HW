import os
import json
import glob
from PIL import Image
from tqdm import tqdm

def yolo_to_coco(yolo_dataset_path, output_json_path, image_ext='.jpg'):
    """
    Convert YOLO format annotations to COCO format
    
    Args:
        yolo_dataset_path: Path to the YOLO dataset directory
        output_json_path: Path where to save the COCO JSON file
        image_ext: Image extension (default: .jpg)
    """
    # Initialize COCO format structure
    coco_format = {
        "info": {
            "description": "Converted from YOLO format",
            "url": "",
            "version": "1.0",
            "year": 2023,
            "contributor": "Converted using YOLO to COCO script",
            "date_created": ""
        },
        "licenses": [
            {
                "url": "",
                "id": 1,
                "name": "Unknown"
            }
        ],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Load class names from coco128.yaml or classes.txt if available
    class_file = os.path.join(yolo_dataset_path, "coco128.yaml")
    if not os.path.exists(class_file):
        class_file = os.path.join(yolo_dataset_path, "classes.txt")
    
    if os.path.exists(class_file):
        if class_file.endswith('.yaml'):
            with open(class_file, 'r') as f:
                content = f.read()
                # Try to extract the class names from YAML
                try:
                    # Simple extraction - assumes format like "names: ['person', 'bicycle', ...]"
                    names_start = content.find("names:")
                    if names_start != -1:
                        names_content = content[names_start:].split("\n")[0]
                        class_names = eval(names_content.split("names:")[1].strip())
                    else:
                        # Fallback to standard COCO classes
                        class_names = get_coco_classes()
                except:
                    class_names = get_coco_classes()
        else:
            # Read from classes.txt
            with open(class_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
    else:
        # Fallback to standard COCO classes
        class_names = get_coco_classes()
    
    # Build categories
    for i, class_name in enumerate(class_names):
        coco_format["categories"].append({
            "supercategory": "none",
            "id": i,
            "name": class_name
        })
    
    # Find images and labels
    image_paths = glob.glob(os.path.join(yolo_dataset_path, 'images', '**', f'*{image_ext}'), recursive=True)
    if not image_paths:  # Try without 'images' subdirectory
        image_paths = glob.glob(os.path.join(yolo_dataset_path, f'**/*{image_ext}'), recursive=True)
    
    if not image_paths:
        raise Exception(f"No images found with extension {image_ext}")
        
    print(f"Found {len(image_paths)} images")
    
    annotation_id = 1
    
    # Process each image and its annotations
    for image_id, image_path in enumerate(tqdm(image_paths, desc="Converting annotations")):
        # Get image dimensions
        img = Image.open(image_path)
        width, height = img.size
        
        # Add image info
        file_name = os.path.basename(image_path)
        coco_format["images"].append({
            "id": image_id,
            "license": 1,
            "file_name": file_name,
            "height": height,
            "width": width,
            "date_captured": ""
        })
        
        # Find corresponding label file
        label_path = image_path.replace('images', 'labels').replace(image_ext, '.txt')
        if not os.path.exists(label_path):
            # Try alternative path patterns
            base_name = os.path.splitext(file_name)[0]
            possible_paths = [
                os.path.join(os.path.dirname(image_path).replace('images', 'labels'), base_name + '.txt'),
                os.path.join(yolo_dataset_path, 'labels', base_name + '.txt'),
                os.path.join(os.path.dirname(image_path), base_name + '.txt')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    label_path = path
                    break
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class_id, x_center, y_center, width, height
                        class_id = int(parts[0])
                        
                        # YOLO format: normalized [x_center, y_center, width, height]
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        box_width = float(parts[3])
                        box_height = float(parts[4])
                        
                        # Convert to COCO format: [x_min, y_min, width, height]
                        x_min = (x_center - box_width / 2) * width
                        y_min = (y_center - box_height / 2) * height
                        box_width_px = box_width * width
                        box_height_px = box_height * height
                        
                        # Create COCO annotation
                        coco_format["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [x_min, y_min, box_width_px, box_height_px],
                            "area": box_width_px * box_height_px,
                            "segmentation": [],
                            "iscrowd": 0
                        })
                        
                        annotation_id += 1
    
    # Save the COCO format JSON
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f, indent=2)
    
    print(f"Conversion complete. COCO format annotations saved to {output_json_path}")
    print(f"Total images: {len(coco_format['images'])}")
    print(f"Total annotations: {len(coco_format['annotations'])}")

def get_coco_classes():
    """Return the standard COCO dataset classes"""
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

if __name__ == "__main__":
    input_path = "datasets/coco128/images/train2017"
    output_path = "datasets/coco128/annotations/instances_train2017v2.json"
    image_ext = ".jpg"
    
    yolo_to_coco(input_path, output_path, image_ext)