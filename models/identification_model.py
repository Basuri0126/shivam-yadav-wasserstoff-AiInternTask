import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import os
import json

# Load a pre-trained Faster R-CNN model
def get_identification_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

# Function to identify objects in an image
def identify_objects(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract information
    boxes = outputs[0]['boxes']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    # Filter out low confidence scores
    threshold = 0.5
    boxes = boxes[scores > threshold]
    labels = labels[scores > threshold]
    scores = scores[scores > threshold]

    return boxes, labels, scores

# Function to map label IDs to label names
def get_label_names():
    # Load COCO labels
    with open("coco_labels.txt") as f:
        labels = f.readlines()
    return [label.strip() for label in labels]

# Function to process segmented objects and identify them
def process_segmented_objects(segmented_objects_dir, output_dir, model):
    label_names = get_label_names()
    metadata_files = [f for f in os.listdir(segmented_objects_dir) if f.endswith('_metadata.json')]

    for metadata_file in metadata_files:
        with open(os.path.join(segmented_objects_dir, metadata_file)) as f:
            metadata = json.load(f)

        for obj in metadata['objects']:
            image_path = obj['image_path']
            boxes, labels, scores = identify_objects(model, image_path)
            
            obj['identification'] = []
            for box, label, score in zip(boxes, labels, scores):
                if label.item() < len(label_names):
                    obj['identification'].append({
                        'label': label_names[label.item()],
                        'score': score.item(),
                        'bbox': box.tolist()
                    })
                else:
                    obj['identification'].append({
                        'label': f"unknown_{label.item()}",
                        'score': score.item(),
                        'bbox': box.tolist()
                    })

        # Save updated metadata
        output_metadata_file = os.path.join(output_dir, metadata_file)
        with open(output_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    model = get_identification_model()
    segmented_objects_dir = r'C:\Users\HP\Desktop\pipline\data\segmented_objects'
    output_dir = r'C:\Users\HP\Desktop\pipline\data\output'

    os.makedirs(output_dir, exist_ok=True)
    process_segmented_objects(segmented_objects_dir, output_dir, model)

    print("Object identification completed successfully.")
