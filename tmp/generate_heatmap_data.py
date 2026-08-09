
import csv
import json
from pathlib import Path
from collections import Counter
import re

def generate_heatmap_data():
    """
    Generates data for a heatmap of road_type vs. dataset split (train/val/test).
    """
    base_path = Path('d:/Jack_Projects/Dataset traffic/emergency-lane-dataset-pipeline/data_collection/dataset_output/dataset-v0.1/PILOT_500_UA_ONLY')
    metadata_path = Path('d:/Jack_Projects/Dataset traffic/emergency-lane-dataset-pipeline/data_collection/reports/external_eda/sequence_scene_metadata.csv')
    output_path = Path('d:/Jack_Projects/Dataset traffic/emergency-lane-dataset-pipeline/tmp/heatmap_data.json')

    # 1. Read metadata
    metadata = {}
    with open(metadata_path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle potential variations in sequence_id format
            seq_id = row.get('sequence_id', '').replace('MVI_', '')
            if seq_id:
                metadata[seq_id] = row

    # 2. Find image files and categorize them
    splits = ['train', 'val', 'test']
    image_data = []
    for split in splits:
        split_path = base_path / 'images' / ('cross_test' if split == 'test' else split)
        for image_file in split_path.glob('*.jpg'):
            # Extract sequence_id from filename, e.g., UA_MVI_20011_00067.jpg -> 20011
            match = re.search(r'MVI_(\d+)', image_file.name)
            if match:
                sequence_id = match.group(1)
                image_data.append({'split': split, 'sequence_id': sequence_id, 'file': image_file.name})

    # 3. Join image data with metadata and aggregate
    heatmap_counts = Counter()
    for item in image_data:
        meta_info = metadata.get(item['sequence_id'])
        if meta_info:
            road_type = meta_info.get('road_type', 'UNKNOWN')
            heatmap_counts[(road_type, item['split'])] += 1
        else:
            heatmap_counts[('UNKNOWN', item['split'])] += 1
            
    # 4. Format for D3.js
    # d3.js heatmap works well with an array of objects
    # like {x: "train", y: "HIGHWAY", value: 123}
    formatted_data = []
    for (road_type, split), count in heatmap_counts.items():
        formatted_data.append({'x': split, 'y': road_type, 'value': count})
        
    # 5. Save to JSON
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=2)

    print(f"Heatmap data saved to {output_path}")

if __name__ == '__main__':
    generate_heatmap_data()
