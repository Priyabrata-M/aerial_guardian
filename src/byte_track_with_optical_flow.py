import cv2
import numpy as np
from pathlib import Path
import time

np.float = float
def estimate_camera_motion(prev_frame, curr_frame):
    """
    Estimate camera motion using optical flow
    """


    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Get median flow (robust to moving objects)
    median_flow_x = np.median(flow[..., 0])
    median_flow_y = np.median(flow[..., 1])

    return np.array([median_flow_x, median_flow_y])

def process_video_with_motion_compensation(
    sequence_path,
    output_path,
    max_frames=None
):
    """
    Process video with tracking + camera motion compensation
    """

    # Get all frames
    frame_files = sorted(Path(sequence_path).glob('*.jpg'))
    if max_frames:
        frame_files = frame_files[:max_frames]

    print("="*70)
    print("PROCESSING VIDEO WITH MOTION COMPENSATION")
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
        30.0,
        (width, height)
    )

    # Reset tracker
    tracker = BYTETracker(Args())

    # Motion compensation variables
    prev_frame = None
    camera_motion = np.array([0.0, 0.0])

    # Process each frame
    frame_times = []
    total_tracks = 0
    motion_magnitudes = []

    for i, frame_path in enumerate(frame_files, 1):
        start_time = time.time()

        # Read frame
        frame = cv2.imread(str(frame_path))

        # Estimate camera motion
        if prev_frame is not None:
            camera_motion = estimate_camera_motion(prev_frame, frame)
            motion_magnitudes.append(np.linalg.norm(camera_motion))

        # Detect
        results = model.predict(
            frame,
            conf=0.20,
            imgsz=1280,
            classes=[0],
            verbose=False
        )[0]

        # Convert detections to ByteTrack format with motion compensation
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()

            # Compensate for camera motion
            x1_comp = x1 - camera_motion[0]
            y1_comp = y1 - camera_motion[1]
            x2_comp = x2 - camera_motion[0]
            y2_comp = y2 - camera_motion[1]

            detections.append([x1_comp, y1_comp, x2_comp, y2_comp, conf])

        # Update tracker with compensated detections
        if len(detections) > 0:
            detections = np.array(detections)
            online_targets = tracker.update(
                detections,
                [height, width],
                [height, width]
            )
        else:
            online_targets = []

        # Draw tracked objects (compensate back to original frame)
        for track in online_targets:
            tlbr = track.tlbr
            track_id = track.track_id

            # Compensate back to current frame coordinates
            x1 = int(tlbr[0] + camera_motion[0])
            y1 = int(tlbr[1] + camera_motion[1])
            x2 = int(tlbr[2] + camera_motion[0])
            y2 = int(tlbr[3] + camera_motion[1])

            # Draw bounding box
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
        motion_mag = np.linalg.norm(camera_motion)
        info_text = f"Frame: {i}/{len(frame_files)} | Tracks: {len(online_targets)} | Motion: {motion_mag:.1f}px"
        cv2.putText(
            frame, info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Write frame
        out_video.write(frame)

        # Update for next iteration
        prev_frame = frame.copy()

        # Calculate FPS
        frame_time = time.time() - start_time
        frame_times.append(frame_time)
        total_tracks += len(online_targets)

        # Print progress
        if i % 50 == 0:
            avg_fps = 1.0 / np.mean(frame_times[-50:])
            avg_motion = np.mean(motion_magnitudes[-50:]) if motion_magnitudes else 0
            print(f"Frame {i}/{len(frame_files)} | Tracks: {len(online_targets)} | FPS: {avg_fps:.2f} | Avg Motion: {avg_motion:.1f}px")

    out_video.release()

    # Summary
    avg_time = np.mean(frame_times)
    avg_fps = 1.0 / avg_time
    avg_motion = np.mean(motion_magnitudes) if motion_magnitudes else 0

    print("\n" + "="*70)
    print("TRACKING WITH MOTION COMPENSATION COMPLETE")
    print("="*70)
    print(f"Output video: {output_path}")
    print(f"Total frames: {len(frame_files)}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average tracks per frame: {total_tracks / len(frame_files):.1f}")
    print(f"Average camera motion: {avg_motion:.2f} pixels")
    print("="*70)

    return avg_fps

# Process with motion compensation
sequence_path = '/content/drive/MyDrive/botlab/VisDrone2019-MOT-train/sequences/uav0000013_01073_v'
output_path = '/content/drive/MyDrive/botlab/outputs/tracked_video_uav0000013_01073_v.mp4'

avg_fps = process_video_with_motion_compensation(
    sequence_path,
    output_path,
    max_frames=200
)