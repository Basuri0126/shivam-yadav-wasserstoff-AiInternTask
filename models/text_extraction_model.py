import easyocr
from PIL import Image
import json
import os

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

if __name__ == "__main__":
    segmented_objects_dir = r'C:\Users\HP\Desktop\pipline\data\segmented_objects'
    output_dir = r'C:\Users\HP\Desktop\pipline\data\output'

    os.makedirs(output_dir, exist_ok=True)
    process_segmented_objects_for_text(segmented_objects_dir, output_dir)

    print("Text extraction completed successfully.")
