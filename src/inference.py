#Inference

# scripts/test_detection.py

import cv2
from ultralytics import YOLO
import os
from pathlib import Path

def test_yolo_on_visdrone(
    model_name='yolo11s.pt',
    sequence_name='uav0000009_03358_v',
    num_frames=10,
    conf_threshold=0.3
):
    """
    Test YOLO detection on VisDrone frames
    """

    # Load model
    print(f"Loading {model_name}...")
    model = YOLO(model_name)

    # Path to sequence
    seq_path = Path(f"/content/drive/MyDrive/botlab/VisDrone2019-MOT-val/sequences/{sequence_name}")
    output_path = Path(f"/content/drive/MyDrive/botlab/VisDrone2019-MOT-val/detection_test_{sequence_name}")
    output_path.mkdir(parents=True, exist_ok=True)

    # Get frames
    frames = sorted(seq_path.glob("*.jpg"))[:num_frames]

    print(f"Processing {len(frames)} frames from {sequence_name}...")

    for frame_path in frames:
        # Read frame
        img = cv2.imread(str(frame_path))

        # Run detection (filter for person classes: 0 in COCO = person)
        # Note: YOLO COCO class 0 = person, VisDrone class 1,2 = person
        results = model.predict(
            img,
            conf=conf_threshold,
            classes=[0],  # person class in COCO
            verbose=False
        )[0]

        # Draw results
        annotated = results.plot()

        # Count detections
        num_detections = len(results.boxes)

        # Add text overlay
        cv2.putText(
            annotated,
            f"Detections: {num_detections} | Conf: {conf_threshold}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Save
        output_file = output_path / frame_path.name
        cv2.imwrite(str(output_file), annotated)

        print(f"  {frame_path.name}: {num_detections} persons detected")

    print(f"\nResults saved to: {output_path}")
    print(f"Check the images to see detection quality!")

if __name__ == "__main__":
    # Test with YOLOv8s (small model)
    test_yolo_on_visdrone(
        model_name='yolo11s.pt',
        sequence_name='uav0000086_00000_v',  # Change to match your sequence
        num_frames=20,
        conf_threshold=0.25
    )