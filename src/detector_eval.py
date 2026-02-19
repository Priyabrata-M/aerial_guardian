from ultralytics import YOLO

# Load model
model = YOLO('/content/drive/MyDrive/botlab/visdrone_training/yolo_all_humans/weights/best.pt')

# Run validation evaluation
print("="*70)
print("Running Validation Evaluation...")
print("="*70)

results = model.val(
    data='/content/drive/MyDrive/botlab/VisDrone-YOLO-Test/data.yaml',
    imgsz=1280,
    batch=8,
    conf=0.001,  # Low threshold to calculate metrics properly
    iou=0.6,
    device=0,
    plots=True,
    save_json=True,
    project='/content/drive/MyDrive/botlab/outputs_uav0000013_01073_v',
    name='validation_metrics'
)

# Print key metrics
print("\n" + "="*70)
print("VALIDATION METRICS")
print("="*70)
print(f"mAP@0.5:     {results.box.map50:.4f}")
print(f"mAP@0.5-0.95: {results.box.map:.4f}")
print(f"Precision:   {results.box.mp:.4f}")
print(f"Recall:      {results.box.mr:.4f}")
print("="*70)