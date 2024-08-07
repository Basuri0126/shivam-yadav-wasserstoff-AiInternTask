import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os
import json
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torch
from torchvision.transforms import functional as F
import numpy as np
import cv2
import easyocr
import datetime
import uuid
import pandas as pd

# Function to draw annotations on the image
def draw_annotations(image, annotations):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for ann in annotations:
        bbox = ann['bbox']
        label = ann['label']
        draw.rectangle(bbox, outline='red', width=2)
        draw.text((bbox[0], bbox[1] - 10), label, fill='red', font=font)

    return image

# Function to generate the final output
def generate_output(data_mapping, input_image_path, output_image_path, output_table_path):
    image = Image.open(input_image_path).convert("RGB")

    annotations = []
    for obj in data_mapping:
        label = f"ID: {obj['unique_id']}\nDesc: {obj['description']}"
        bbox = obj.get('bbox', [10, 10, 100, 100])  # Dummy bounding box
        annotations.append({'bbox': bbox, 'label': label})

    annotated_image = draw_annotations(image, annotations)
    annotated_image.save(output_image_path)

    # Create a table of all mapped data
    table_content = "Master ID,Unique ID,Description,Text,Summary\n"
    for obj in data_mapping:
        row = f"{obj['master_id']},{obj['unique_id']},{obj['description']},{obj['text']},{obj['summary']}\n"
        table_content += row

    with open(output_table_path, 'w') as f:
        f.write(table_content)

    print("Output generation completed successfully.")

# Function to map data to each object
def map_data_to_objects(segmented_objects_dir, output_dir):
    metadata_files = [f for f in os.listdir(segmented_objects_dir) if f.endswith('_metadata.json')]

    data_mapping = []
    for metadata_file in metadata_files:
        with open(os.path.join(segmented_objects_dir, metadata_file)) as f:
            metadata = json.load(f)

        master_id = metadata['master_id']
        for obj in metadata['objects']:
            unique_id = obj['unique_id']
            description = obj.get('description', '')
            text = obj.get('text', '')
            summary = obj.get('summary', '')

            data_entry = {
                'master_id': master_id,
                'unique_id': unique_id,
                'description': description,
                'text': text,
                'summary': summary
            }
            data_mapping.append(data_entry)

    # Save the data mapping as a JSON file
    output_mapping_file = os.path.join(output_dir, 'data_mapping.json')
    with open(output_mapping_file, 'w') as f:
        json.dump(data_mapping, f, indent=4)
    
    return data_mapping

# Function to summarize the attributes of each object
def summarize_attributes(metadata):
    for obj in metadata['objects']:
        description = obj.get('description', '')
        text = obj.get('text', '')
        summary = f"Description: {description}\nText: {text}"
        obj['summary'] = summary

    return metadata

# Function to process segmented objects and summarize attributes
def process_segmented_objects_for_summary(segmented_objects_dir, output_dir):
    metadata_files = [f for f in os.listdir(segmented_objects_dir) if f.endswith('_metadata.json')]

    for metadata_file in metadata_files:
        with open(os.path.join(segmented_objects_dir, metadata_file)) as f:
            metadata = json.load(f)

        metadata = summarize_attributes(metadata)

        # Save updated metadata
        output_metadata_file = os.path.join(output_dir, metadata_file)
        with open(output_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

# Function to extract text from an image using EasyOCR
def extract_text(image_path):
    reader = easyocr.Reader(['en'])  # Specify the language(s) you want to use
    result = reader.readtext(image_path, detail=0)  # Set detail=0 to get plain text
    text = ' '.join(result)
    return text

# Function to process segmented objects and extract text
def process_segmented_objects_for_text(segmented_objects_dir, output_dir):
    metadata_files = [f for f in os.listdir(segmented_objects_dir) if f.endswith('_metadata.json')]

    for metadata_file in metadata_files:
        with open(os.path.join(segmented_objects_dir, metadata_file)) as f:
            metadata = json.load(f)

        for obj in metadata['objects']:
            image_path = obj['image_path']
            text = extract_text(image_path)
            obj['text'] = text

        # Save updated metadata
        output_metadata_file = os.path.join(output_dir, metadata_file)
        with open(output_metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

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

    segmented_image = Image.fromarray(image)
    return segmented_image

# Define directories
segmented_objects_dir = os.path.join("data", "segmented_objects")
output_dir = os.path.join("data", "output")
input_images_dir = os.path.join("data", "input_images")

# Ensure directories exist
os.makedirs(segmented_objects_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(input_images_dir, exist_ok=True)

# Streamlit app
st.title("AI Pipeline for Image Segmentation and Object Analysis")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Generate a unique identifier for the current image
    unique_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_dir_name = f"{timestamp}_{unique_id}"

    # Create a unique directory for the current image upload within the output directory
    unique_dir_path = os.path.join(output_dir, unique_dir_name)
    os.makedirs(unique_dir_path, exist_ok=True)

    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Segmenting image...")

    # Segment the image
    segmentation_model = get_segmentation_model()
    segmented_image = segment_image(segmentation_model, uploaded_file)

    # Display the segmented image
    st.image(segmented_image, caption="Segmented Image.", use_column_width=True)

    # Save the segmented image in the unique directory
    segmented_image_path = os.path.join(unique_dir_path, "segmented_image.png")
    segmented_image.save(segmented_image_path)
    st.write(f"Segmented image saved to {segmented_image_path}")

    # Define directories for segmented objects
    segmented_objects_dir = os.path.join(unique_dir_path, "segmented_objects")
    os.makedirs(segmented_objects_dir, exist_ok=True)

    # Process segmented objects for text and summary
    process_segmented_objects_for_text(segmented_objects_dir, segmented_objects_dir)
    process_segmented_objects_for_summary(segmented_objects_dir, segmented_objects_dir)
    data_mapping = map_data_to_objects(segmented_objects_dir, segmented_objects_dir)

    # Display individual segmented objects and their details
    for obj in data_mapping:
        st.write(f"Object ID: {obj['unique_id']}")
        obj_image_path = os.path.join(segmented_objects_dir, f"{obj['unique_id']}.png")
        if os.path.exists(obj_image_path):
            obj_image = Image.open(obj_image_path)
            st.image(obj_image, caption=f"Object ID: {obj['unique_id']}", use_column_width=True)
        st.write(f"Description: {obj['description']}")
        st.write(f"Extracted Text: {obj['text']}")
        st.write(f"Summary: {obj['summary']}")

    # Save mapped data
    mapped_data_path = os.path.join(segmented_objects_dir, "mapped_data.json")
    with open(mapped_data_path, "w") as f:
        json.dump(data_mapping, f)
    st.write(f"Mapped data saved to {mapped_data_path}")

    # Generate final output
    final_output_path = os.path.join(unique_dir_path, "final_output.png")
    output_table_path = os.path.join(unique_dir_path, "output_table.csv")
    generate_output(data_mapping, segmented_image_path, final_output_path, output_table_path)
    st.image(final_output_path, caption="Final Output Image with Annotations.", use_column_width=True)
    st.write(f"Final output saved to {final_output_path}")

    # Display the table of all mapped data
    st.write("Table of Mapped Data:")
    st.write(pd.DataFrame(data_mapping))
