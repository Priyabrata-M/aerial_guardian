# aerial_guardian

Assumption: For this assignment I have considered pedestrian (1), people (2), bicyclist (3), motobike rider(10) all into human class.


All the dependancies are added to requirement.txt and we can install the dependancies with pip install -r requirements.txt 


# Detection

For detection I have used yolo11 and to deal with small objects while detection:

1. High Resolution (imgsz=1280)
I have doubled the input resolution to 1280, this preserve the pixel density of tiny objects, preventing them from being "downsampled" into unrecognizable blobs.

2. I have increased the box loss to 9 forces the model to be much more precise about the bounding box coordinates. For small objects, a shift of just 2–3 pixels can result in a 0% IoU. this setting helps fight that.

3. Mosaic and Scale Augmentation used making the model practice detecting smaller versions of the objects with complex back ground.


This is the dataset I have used for training:
Total sequences: 6

| Sequence               | Frames | Persons | Persons/Frame |
|------------------------|--------|---------|---------------|
| uav0000086_00000_v     | 464    | 22098   | 47.62         |
| uav0000117_02622_v     | 349    | 9670    | 27.71         |
| uav0000137_00458_v     | 233    | 9299    | 39.91         |
| uav0000182_00000_v     | 363    | 1175    | 3.24          |
| uav0000268_05773_v     | 978    | 1984    | 2.03          |
| uav0000305_00000_v     | 184    | 603     | 3.28          |
| **TOTAL**              | **2571** | **44829** | **17.68** |




Human Bounding Box Sizes (area in pixels):
| Metric  | Value |
|---------|-------|
| Min     | 96    |
| Max     | 15041 |
| Mean    | 2955  |
| Median  | 2233  |


I have finetunned for 80 epochs.

Model evaluated on uav0000339_00001_v sequence.
Evaluation metircs for detection:

## Test Metrics

| Metric          | Value   |
|-----------------|---------|
| mAP@0.5         | 0.8388  |
| mAP@0.5–0.95    | 0.4371  |
| Precision       | 0.8056  |
| Recall          | 0.8123  |
| Average FPS     | 36.75   |
| Min FPS         | 30.54   |
| Max FPS         | 40.29   |


Model checkpoints: /aerial_guardian/check_points/weights/best.pt

# Tracking:

I have tried to do tracking with ByteTrack, ByteTrack with optical flow for motion compensation and DeepSORT.


# Install ByteTrack
pip install cython cython-bbox
pip install lap  # This often fails if build-essential is missing

Clone the repository
git clone https://github.com/ifzhang/ByteTrack.git

Move into the directory
cd ByteTrack

Install the specific requirements for ByteTrack
pip install -r requirements.txt

Build the project (This compiles the C++ extensions)
python setup.py develop





With ByteTrack:
Output video: aerial_guardian/output/tracked_video_final.mp4
Average FPS: 17.06

## Key Metrics Summary — ByteTrack

| Metric | Value |
|--------|-------|
| MOTA   | 55.00%|
| MOTP   | 7.18% |
| IDF1   | 70.87%|
| Precision | 95.82% |
| Recall | 57.92% |
| ID Switches | 4 |
| Fragmentations | 7 |




To address ID switching caused by drone ego-motion or occlusions:

I implemented camera motion compensation using optical flow to estimate global camera movement between frames. Track predictions are adjusted by the inverse camera motion before matching with new detections, significantly reducing false ID switches caused by drone movement.

For occlusions, I increased the track memory buffer from 30 to 90 frames, allowing tracks to persist longer when persons temporarily disappear behind obstacles. 

With ByteTrack with MOTION COMPENSATION:
Average FPS: 5.14
Average camera motion: 0.23 pixels



## Key Metrics Summary — ByteTrack + Optical Flow

### Tracking Performance

| Metric | Value |
|--------|-------|
| MOTA   | 55.00% |
| IDF1   | 70.87% |
| Precision | 95.82% |
| Recall | 57.92% |

### Tracking Stability

| Metric | Value |
|--------|-------|
| ID Switches | 4 |
| Fragmentations | 7 |

### Motion Analysis

| Metric | Value |
|--------|-------|
| Average Camera Motion | 0.18 pixels |

Output video: aerial_guardian/output/tracked_video_motion_comp.mp4


For sequences with camera motion below 1-2 pixels/frame, motion compensation overhead is not justified. Our baseline ByteTrack tracker achieved excellent results (55.00% MOTA, 70.87% IDF1, only 4 ID switches across 58 frames) without motion compensation. Camera motion compensation should be reserved for sequences with significant ego-motion (>5 pixels/frame), such as aggressive drone maneuvering or tracking scenarios.


# Install DeepSORT dependencies
!pip install filterpy scikit-image
!pip install gdown

# Clone DeepSORT repository
!git clone https://github.com/nwojke/deep_sort.git
%cd deep_sort


Additionally I have tested the sequence for DeepSort with appearance-based re-identification features but there is no major improvement in result but the FPS was very low compared to ByteSort and as per my understanding due to small object size deep sort appearance features are not much useful here. 


# edge deployment

The model is already light weight i.e. 25MB compared to our budget of 500MB but still if we need ligher then we can go for prunning and quantization.
we can use export_model.py to convert pt file into onnx.
python scripts/export_model.py --model checkpoints/weights/best.pt 
I have also added onnx file the checkpoint folder. 

Convert to TensorRT (on Jetson):
trtexec --onnx=best.onnx --saveEngine=best.trt --fp16
  



