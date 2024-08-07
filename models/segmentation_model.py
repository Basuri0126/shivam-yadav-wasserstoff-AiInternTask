import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json

# Load a pre-trained Mask R-CNN model
def get_segmentation_model():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

# Function to segment objects in an image
def segment_image(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract the masks
    masks = outputs[0]['masks']
    labels = outputs[0]['labels']
    scores = outputs[0]['scores']

    # Filter out low confidence scores
    threshold = 0.5
    masks = masks[scores > threshold]
    labels = labels[scores > threshold]
    scores = scores[scores > threshold]

    # Visualize the masks on the image
    result = visualize_segmentation(image, masks, labels)

    return masks, labels, scores, result

# Function to visualize segmentation masks
def visualize_segmentation(image, masks, labels):
    image = np.array(image)
    for i in range(masks.shape[0]):
        mask = masks[i, 0].mul(255).byte().cpu().numpy()
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8)
        color = [int(c) for c in color[0]]
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, color, 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    return image

# Function to extract and save segmented objects
def extract_and_save_objects(image_path, masks, labels, scores, output_dir):
    image = Image.open(image_path).convert("RGB")
    master_id = os.path.basename(image_path).split('.')[0]

    # Metadata to store
    metadata = {
        "master_id": master_id,
        "objects": []
    }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    for idx, (mask, label, score) in enumerate(zip(masks, labels, scores)):
        unique_id = f"{master_id}_obj_{idx}"
        
        # Extract the mask
        mask = mask[0].mul(255).byte().cpu().numpy()
        mask_image = Image.fromarray(mask)

        # Extract the object using the mask
        object_image = Image.composite(image, Image.new("RGB", image.size), mask_image)

        # Find bounding box coordinates
        mask_np = np.array(mask)
        coords = cv2.findNonZero(mask_np)
        x, y, w, h = cv2.boundingRect(coords)
        
        # Crop to bounding box
        object_image = object_image.crop((x, y, x+w, y+h))

        # Save the object image
        object_image_path = os.path.join(output_dir, f"{unique_id}.png")
        object_image.save(object_image_path)

        # Add metadata
        metadata["objects"].append({
            "unique_id": unique_id,
            "label": label.item(),
            "score": score.item(),
            "bbox": [x, y, w, h],
            "image_path": object_image_path
        })

    # Save metadata as JSON
    with open(os.path.join(output_dir, f"{master_id}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    model = get_segmentation_model()
    image_path = r'C:\Users\HP\Desktop\pipline\data\input_images\test.PNG'
    output_dir = r'C:\Users\HP\Desktop\pipline\data\segmented_objects'

    masks, labels, scores, segmented_image = segment_image(model, image_path)
    extract_and_save_objects(image_path, masks, labels, scores, output_dir)

    print("Object extraction and saving completed successfully.")
