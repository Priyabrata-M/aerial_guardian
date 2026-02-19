import os
from pathlib import Path
import shutil
from tqdm import tqdm
import cv2

# Configuration
visdrone_val_root = '/content/drive/MyDrive/botlab/VisDrone2019-MOT-val'
output_root = '/content/drive/MyDrive/botlab/VisDrone-YOLO-Train'

# Clear old data first
import shutil
if Path(output_root).exists():
    shutil.rmtree(output_root)
    print("🗑️  Cleared old dataset")

sequences_dir = Path(visdrone_val_root) / 'sequences'
annotations_dir = Path(visdrone_val_root) / 'annotations'

train_sequences = [
    'uav0000086_00000_v',
    'uav0000117_02622_v',
    'uav0000137_00458_v',
    'uav0000182_00000_v',
    'uav0000268_05773_v',
    'uav0000305_00000_v',
    'uav0000339_00001_v'
]

val_sequences = ['uav0000339_00001_v']

print("="*70)
print("VisDrone to YOLO Conversion - ALL HUMAN CLASSES")
print("="*70)
print("Including: pedestrian (1), people (2), bicycle (3), motor (10)")
print(f"Training sequences: {len(train_sequences)}")
print("="*70)

def convert_sequences(sequence_list, split_name):
    """Convert sequences to YOLO format - ALL HUMANS"""

    output_images = Path(output_root) / split_name / 'images'
    output_labels = Path(output_root) / split_name / 'labels'

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_annotations = 0
    class_counts = {1: 0, 2: 0, 3: 0, 10: 0}  # Track each class

    for seq_name in tqdm(sequence_list, desc=f"Converting {split_name}"):
        seq_path = sequences_dir / seq_name
        anno_file = annotations_dir / f"{seq_name}.txt"

        if not anno_file.exists():
            print(f"⚠️  No annotation for {seq_name}")
            continue

        # Read annotations
        frame_annotations = {}
        with open(anno_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue

                frame_idx = int(parts[0])
                bbox_left = int(parts[2])
                bbox_top = int(parts[3])
                bbox_width = int(parts[4])
                bbox_height = int(parts[5])
                category = int(parts[7])

                # ✅ UPDATED: Include ALL human-related classes
                if category not in [1, 2, 3, 10]:
                    continue

                class_counts[category] += 1

                if frame_idx not in frame_annotations:
                    frame_annotations[frame_idx] = []

                frame_annotations[frame_idx].append({
                    'bbox': [bbox_left, bbox_top, bbox_width, bbox_height]
                })

        # Process frames
        for img_file in sorted(seq_path.glob("*.jpg")):
            frame_idx = int(img_file.stem)

            img = cv2.imread(str(img_file))
            if img is None:
                continue

            img_height, img_width = img.shape[:2]

            # Copy image
            new_img_name = f"{seq_name}_{img_file.name}"
            shutil.copy(img_file, output_images / new_img_name)

            # Create label
            label_file = output_labels / f"{seq_name}_{img_file.stem}.txt"

            if frame_idx in frame_annotations:
                with open(label_file, 'w') as f:
                    for anno in frame_annotations[frame_idx]:
                        bbox = anno['bbox']

                        x_center = (bbox[0] + bbox[2] / 2) / img_width
                        y_center = (bbox[1] + bbox[3] / 2) / img_height
                        width = bbox[2] / img_width
                        height = bbox[3] / img_height

                        x_center = max(0, min(1, x_center))
                        y_center = max(0, min(1, y_center))
                        width = max(0, min(1, width))
                        height = max(0, min(1, height))

                        f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        total_annotations += 1

                total_images += 1
            else:
                label_file.touch()
                total_images += 1

    print(f"\n  {split_name}: {total_images} images, {total_annotations} annotations")
    print(f"  Class breakdown:")
    print(f"    - Pedestrian (1): {class_counts[1]}")
    print(f"    - People (2):     {class_counts[2]}")
    print(f"    - Bicycle (3):    {class_counts[3]}")
    print(f"    - Motor (10):     {class_counts[10]}")

    return total_images, total_annotations

# Convert
train_imgs, train_annos = convert_sequences(train_sequences, 'train')
val_imgs, val_annos = convert_sequences(val_sequences, 'val')

print("\n" + "="*70)
print("✅ Conversion Complete - ALL HUMAN CLASSES")
print("="*70)
print(f"TRAIN: {train_imgs} images, {train_annos} annotations")
print(f"VAL:   {val_imgs} images, {val_annos} annotations")
print("="*70)