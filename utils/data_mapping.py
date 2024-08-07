import json
import os

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

if __name__ == "__main__":
    segmented_objects_dir = r'C:\Users\HP\Desktop\pipline\data\segmented_objects'
    output_dir = r'C:\Users\HP\Desktop\pipline\data\output'

    os.makedirs(output_dir, exist_ok=True)
    map_data_to_objects(segmented_objects_dir, output_dir)

    print("Data mapping completed successfully.")
