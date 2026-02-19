from ultralytics import YOLO
import torch

print("="*70)
print("YOLOv11 Fine-tuning - ALL HUMAN CLASSES")
print("="*70)
print(f"Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Training: 2,846 images, 68,359 annotations")
print(f"Classes: pedestrian, people, bicycle, motor → all as 'person'")
print("="*70)

model = YOLO('yolo11s.pt')

results = model.train(
    data='/content/drive/MyDrive/botlab/config/data.yaml',

    epochs=80,
    patience=15,

    imgsz=1280,
    batch=-1,  # Auto-batch or use 2 for safety

    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3.0,

    # Augmentation
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.1,
    close_mosaic=10,  # NEW: Disable mosaic in last 10 epochs
    scale=0.5,
    flipud=0.0,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    # Loss weights
    box=7.5,
    cls=0.5,
    dfl=1.5,

    project='/content/drive/MyDrive/botlab/visdrone_training',
    name='yolo11s_all_humans',
    exist_ok=False,

    val=True,
    save=True,
    save_period=10,
    plots=True,

    device=0 if torch.cuda.is_available() else 'cpu',
    workers=4,
    cache='ram',  # Use RAM caching if available

    seed=42,
    deterministic=True,
    verbose=True,
)

print("\n" + "="*70)
print("✅ Training Complete!")
print("="*70)
print(f"Best model: {results.save_dir}/weights/best.pt")
print("="*70)