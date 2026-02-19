from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
from pathlib import Path
import time

# Initialize DeepSORT with deep-sort-realtime
deepsort_tracker = DeepSort(
    max_age=90,              # Same as ByteTrack for comparison
    n_init=3,                # Frames to confirm track
    max_iou_distance=0.7,    # IOU threshold
    max_cosine_distance=0.4, # Appearance similarity threshold
    embedder="mobilenet",    # Lightweight ReID model
    half=True,               # FP16 for speed
    bgr=True,                # OpenCV uses BGR
    embedder_gpu=True        # Use GPU for ReID
)

print("✅ DeepSORT tracker initialized!")
print(f"Max age: 90 frames")
print(f"ReID model: MobileNet")
print(f"Max cosine distance: 0.4")



def process_video_with_deepsort(
    sequence_path,
    output_path,
    max_frames=None,
    trail_length=30
):
    """
    Process video with DeepSORT tracking + trajectory trails
    """

    # Get all frames
    frame_files = sorted(Path(sequence_path).glob('*.jpg'))
    if max_frames:
        frame_files = frame_files[:max_frames]

    print("="*70)
    print("PROCESSING VIDEO WITH DEEPSORT")
    print("="*70)
    print(f"Sequence: {Path(sequence_path).name}")
    print(f"Total frames: {len(frame_files)}")
    print(f"Trail length: {trail_length} frames")
    print("="*70)

    # Prepare output video
    first_frame = cv2.imread(str(frame_files[0]))
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(
        output_path,
        fourcc,
        30.0,
        (width, height)
    )

    # Reset tracker
    deepsort_tracker = DeepSort(
        max_age=90,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.4,
        embedder="mobilenet",
        half=True,
        bgr=True,
        embedder_gpu=True
    )

    # Store trajectories
    trajectories = {}

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

        # Convert detections to DeepSORT format
        # Format: ([x1, y1, w, h], confidence, class)
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()

            # Convert to [x1, y1, w, h]
            w = x2 - x1
            h = y2 - y1

            detections.append(([x1, y1, w, h], conf, 'person'))

        # Update tracker
        tracks = deepsort_tracker.update_tracks(detections, frame=frame)

        # Update trajectories and draw
        current_ids = set()
        active_tracks = 0

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]

            current_ids.add(track_id)
            active_tracks += 1

            # Calculate center point
            center_x = int((ltrb[0] + ltrb[2]) / 2)
            center_y = int((ltrb[1] + ltrb[3]) / 2)

            # Update trajectory
            if track_id not in trajectories:
                trajectories[track_id] = []
            trajectories[track_id].append((center_x, center_y))

            # Keep only last N points
            if len(trajectories[track_id]) > trail_length:
                trajectories[track_id] = trajectories[track_id][-trail_length:]

            # Draw trajectory trail
            points = trajectories[track_id]
            if len(points) > 1:
                for j in range(1, len(points)):
                    alpha = j / len(points)
                    thickness = max(1, int(3 * alpha))
                    color = (int(255 * (1 - alpha)), int(255 * alpha), 0)

                    cv2.line(frame, points[j-1], points[j], color, thickness)

            # Draw bounding box
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw ID with background
            label = f"ID:{track_id}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                frame, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        # Clean up old trajectories
        trajectories = {k: v for k, v in trajectories.items() if k in current_ids}

        # Add frame info
        info_text = f"Frame: {i}/{len(frame_files)} | Active Tracks: {active_tracks} | DeepSORT"
        cv2.putText(
            frame, info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 255),  # Purple for DeepSORT
            2
        )

        # Write frame
        out_video.write(frame)

        # Calculate FPS
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        total_tracks += active_tracks

        # Print progress
        if i % 50 == 0:
            avg_fps = 1.0 / np.mean(frame_times[-50:])
            print(f"Frame {i}/{len(frame_files)} | Tracks: {active_tracks} | FPS: {avg_fps:.2f}")

    out_video.release()

    # Summary
    avg_time = np.mean(frame_times)
    avg_fps = 1.0 / avg_time

    print("\n" + "="*70)
    print("DEEPSORT TRACKING COMPLETE")
    print("="*70)
    print(f"Output video: {output_path}")
    print(f"Total frames: {len(frame_files)}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average tracks per frame: {total_tracks / len(frame_files):.1f}")
    print("="*70)

    return avg_fps

# Process with DeepSORT
sequence_path = '/content/drive/MyDrive/botlab/VisDrone2019-MOT-val/sequences/uav0000339_00001_v'
output_path = '/content/drive/MyDrive/botlab/outputs/tracked_video_deepsort.mp4'

deepsort_fps = process_video_with_deepsort(
    sequence_path,
    output_path,
    max_frames=275,  # Same as ByteTrack for comparison
    trail_length=30
)