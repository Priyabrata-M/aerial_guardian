import sys
sys.path.insert(0, './ByteTrack')

from yolox.tracker.byte_tracker import BYTETracker
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class Args:
    """Arguments for ByteTracker"""
    track_thresh = 0.5      # Detection confidence threshold for tracking
    track_buffer = 90       # Frames to keep alive (increased for occlusions)
    match_thresh = 0.8      # IOU threshold for matching
    mot20 = False

# Load your fine-tuned model
model = YOLO('/Users/priyabratamallick/Desktop/aerial_guardian/check_points/weights/best.pt')

# Initialize tracker
tracker = BYTETracker(Args())

print("Model and tracker loaded successfully!")
print(f"Track buffer: {Args.track_buffer} frames")
print(f"Track threshold: {Args.track_thresh}")
print(f"Match threshold: {Args.match_thresh}")



import cv2
import numpy as np
from pathlib import Path
import time

import numpy as np
np.float = float

def process_video_with_tracking(
    sequence_path,
    output_path,
    max_frames=None  # Set to number or None for all frames
):
    """
    Process video sequence with detection + tracking
    """

    # Get all frames
    frame_files = sorted(Path(sequence_path).glob('*.jpg'))
    if max_frames:
        frame_files = frame_files[:max_frames]

    print("="*70)
    print("PROCESSING VIDEO WITH TRACKING")
    print("="*70)
    print(f"Sequence: {Path(sequence_path).name}")
    print(f"Total frames: {len(frame_files)}")
    print("="*70)

    # Prepare output video
    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(
        output_path,
        fourcc,
        30.0,  # FPS
        (width, height)
    )

    # Reset tracker
    tracker = BYTETracker(Args())

    # Process each frame
    frame_times = []
    total_tracks = 0

    for i, frame_path in enumerate(frame_files, 1):
        start_time = time.time()

        # Read frame
        frame = cv2.imread(str(frame_path))

        # Detect
        results = model.predict(
            frame,
            conf=0.20,
            imgsz=1280,
            classes=[0],
            verbose=False
        )[0]

        # Convert detections to ByteTrack format
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()

            # ByteTrack format: [x1, y1, x2, y2, score]
            detections.append([x1, y1, x2, y2, conf])

        # Update tracker
        if len(detections) > 0:
            detections = np.array(detections)
            online_targets = tracker.update(
                detections,
                [height, width],
                [height, width]
            )
        else:
            online_targets = []

        # Draw tracked objects
        for track in online_targets:
            tlbr = track.tlbr  # [x1, y1, x2, y2]
            track_id = track.track_id

            # Draw bounding box
            x1, y1, x2, y2 = map(int, tlbr)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw ID
            label = f"ID:{track_id}"
            cv2.putText(
                frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # Add frame info
        info_text = f"Frame: {i}/{len(frame_files)} | Tracks: {len(online_targets)}"
        cv2.putText(
            frame, info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # Write frame
        out_video.write(frame)

        # Calculate FPS
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        total_tracks += len(online_targets)

        # Print progress
        if i % 50 == 0:
            avg_fps = 1.0 / np.mean(frame_times[-50:])
            print(f"Frame {i}/{len(frame_files)} | Tracks: {len(online_targets)} | FPS: {avg_fps:.2f}")

    out_video.release()

    # Summary
    avg_time = np.mean(frame_times)
    avg_fps = 1.0 / avg_time

    print("\n" + "="*70)
    print("TRACKING COMPLETE")
    print("="*70)
    print(f"Output video: {output_path}")
    print(f"Total frames processed: {len(frame_files)}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average tracks per frame: {total_tracks / len(frame_files):.1f}")
    print("="*70)

    return avg_fps

# Process validation sequence
sequence_path = '/Users/priyabratamallick/Desktop/aerial_guardian/output/uav0000076_00720_v'
output_path = '/Users/priyabratamallick/Desktop/aerial_guardian/output/uav0000076_00720_v.mp4'

avg_fps = process_video_with_tracking(
    sequence_path,
    output_path,
    max_frames=200  # Process first 200 frames for testing
)






