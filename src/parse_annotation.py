
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def parse_visdrone_annotation(anno_file):
    """
    Parse VisDrone annotation file
    Format: <frame_id>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,
            <score>,<object_category>,<truncation>,<occlusion>
    """
    data = []
    with open(anno_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    data.append({
                        'frame': int(parts[0]),
                        'id': int(parts[1]),
                        'x': int(parts[2]),
                        'y': int(parts[3]),
                        'w': int(parts[4]),
                        'h': int(parts[5]),
                        'score': int(parts[6]),
                        'category': int(parts[7]),
                        'truncation': int(parts[8]) if len(parts) > 8 else 0,
                        'occlusion': int(parts[9]) if len(parts) > 9 else 0
                    })
    return pd.DataFrame(data)

def analyze_dataset(anno_dir):
    """Analyze VisDrone dataset statistics"""

    category_names = {
        0: 'ignored',
        1: 'pedestrian',
        2: 'people',
        3: 'bicycle',
        4: 'car',
        5: 'van',
        6: 'truck',
        7: 'tricycle',
        8: 'awning-tricycle',
        9: 'bus',
        10: 'motor',
        11: 'others'
    }

    all_categories = []
    all_sizes = []
    person_sizes = []

    for anno_file in os.listdir(anno_dir):
        if anno_file.endswith('.txt'):
            df = parse_visdrone_annotation(os.path.join(anno_dir, anno_file))

            # Collect statistics
            all_categories.extend(df['category'].tolist())
            all_sizes.extend((df['w'] * df['h']).tolist())

            # Person-specific (class 1 and 2)
            person_df = df[df['category'].isin([1, 2])]
            person_sizes.extend((person_df['w'] * person_df['h']).tolist())

    # Print statistics
    print("=" * 60)
    print("VisDrone Dataset Analysis")
    print("=" * 60)

    category_counts = Counter(all_categories)
    print("\nObject Category Distribution:")
    for cat, count in sorted(category_counts.items()):
        name = category_names.get(cat, 'unknown')
        percentage = (count / len(all_categories)) * 100
        print(f"  {cat:2d} - {name:20s}: {count:6d} ({percentage:5.2f}%)")

    # Person statistics
    person_count = category_counts.get(1, 0) + category_counts.get(2, 0)
    print(f"\n{'='*60}")
    print(f"Total Person Objects (class 1+2): {person_count}")
    print(f"Percentage of dataset: {(person_count/len(all_categories))*100:.2f}%")

    if person_sizes:
        import numpy as np
        print(f"\nPerson Bounding Box Sizes (area in pixels):")
        print(f"  Min:    {np.min(person_sizes):.0f}")
        print(f"  Max:    {np.max(person_sizes):.0f}")
        print(f"  Mean:   {np.mean(person_sizes):.0f}")
        print(f"  Median: {np.median(person_sizes):.0f}")

    return all_categories, all_sizes, person_sizes

if __name__ == "__main__":
    anno_dir = "/content/drive/MyDrive/botlab/VisDrone2019-MOT-val/annotations/"
    analyze_dataset(anno_dir)