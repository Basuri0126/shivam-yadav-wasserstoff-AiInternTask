import json
import os

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

if __name__ == "__main__":
    segmented_objects_dir = r'C:\Users\HP\Desktop\pipline\data\segmented_objects'
    output_dir = r'C:\Users\HP\Desktop\pipline\data\output'

    os.makedirs(output_dir, exist_ok=True)
    process_segmented_objects_for_summary(segmented_objects_dir, output_dir)

    print("Attribute summarization completed successfully.")
