import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

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
def generate_output(data_mapping_file, input_image_path, output_image_path, output_table_path):
    with open(data_mapping_file) as f:
        data_mapping = json.load(f)

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

if __name__ == "__main__":
    data_mapping_file = r'C:\Users\HP\Desktop\pipline\data\output\data_mapping.json'
    input_image_path = r'C:\Users\HP\Desktop\pipline\data\input_images\test.PNG'
    output_image_path = r'C:\Users\HP\Desktop\pipline\data\output\annotated_image.png'
    output_table_path = r'C:\Users\HP\Desktop\pipline\data\output\data_table.csv'

    generate_output(data_mapping_file, input_image_path, output_image_path, output_table_path)
