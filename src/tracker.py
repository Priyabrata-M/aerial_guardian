import sys
import argparse
sys.path.insert(0, './ByteTrack')

from yolox.tracker.byte_tracker import BYTETracker
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import time

np.float = float

class Args:
    """Arguments for ByteTracker"""
    track_thresh = 0.5
    track_buffer = 90
    match_thresh = 0.8
    mot20 = False

# Load your fine-tuned model
model = YOLO('aerial_guardian/check_points/weights/best.pt')

# Initialize tracker
tracker = BYTETracker(Args())

print("Model and tracker loaded successfully!")
print(f"Track buffer: {Args.track_buffer} frames")
print(f"Track threshold: {Args.track_thresh}")
print(f"Match threshold: {Args.match_thresh}")


def process_video_with_tracking(sequence_path, output_path, max_frames=None):
    """Process video sequence with detection + tracking"""

    frame_files = sorted(Path(sequence_path).glob('*.jpg'))
    if max_frames:
        frame_files = frame_files[:max_frames]

    print("="*70)
    print("PROCESSING VIDEO WITH TRACKING")
    print("="*70)
    print(f"Sequence: {Path(sequence_path).name}")
    print(f"Total frames: {len(frame_files)}")
    print("="*70)

    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))

    tracker = BYTETracker(Args())

    frame_times = []
    total_tracks = 0

    for i, frame_path in enumerate(frame_files, 1):
        start_time = time.time()

        frame = cv2.imread(str(frame_path))

        results = model.predict(
            frame,
            conf=0.20,
            imgsz=1280,
            classes=[0],
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            detections.append([x1, y1, x2, y2, conf])

        if len(detections) > 0:
            detections = np.array(detections)
            online_targets = tracker.update(detections, [height, width], [height, width])
        else:
            online_targets = []

        for track in online_targets:
            tlbr = track.tlbr
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, tlbr)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        info_text = f"Frame: {i}/{len(frame_files)} | Tracks: {len(online_targets)}"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        out_video.write(frame)

        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        total_tracks += len(online_targets)

        if i % 50 == 0:
            avg_fps = 1.0 / np.mean(frame_times[-50:])
            print(f"Frame {i}/{len(frame_files)} | Tracks: {len(online_targets)} | FPS: {avg_fps:.2f}")

    out_video.release()

    avg_fps = 1.0 / np.mean(frame_times)
    print("\n" + "="*70)
    print("TRACKING COMPLETE")
    print("="*70)
    print(f"Output video: {output_path}")
    print(f"Total frames processed: {len(frame_files)}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average tracks per frame: {total_tracks / len(frame_files):.1f}")
    print("="*70)

    return avg_fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial object detection + tracking with ByteTrack")
    parser.add_argument("--sequence", "-s", required=True,
                        help="Path to folder containing input .jpg frames")
    parser.add_argument("--output", "-o", required=True,
                        help="Path for the output .mp4 video file")
    parser.add_argument("--max-frames", "-m", type=int, default=None,
                        help="Maximum number of frames to process (default: all)")

    args = parser.parse_args()

    process_video_with_tracking(
        sequence_path=args.sequence,
        output_path=args.output,
        max_frames=args.max_frames
    )