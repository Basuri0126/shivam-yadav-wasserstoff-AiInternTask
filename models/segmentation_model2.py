import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

    # Visualize the masks on the image
    result = visualize_segmentation(image, masks, labels)

    return result

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

if __name__ == "__main__":
    model = get_segmentation_model()
    image_path = r'C:\Users\HP\Desktop\pipline\data\input_images\test.PNG'
    segmented_image = segment_image(model, image_path)
