from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Paths
img_path = "/content/drive/MyDrive/botlab/VisDrone2019-MOT-val/sequences/uav0000339_00001_v/0000002.jpg"
ann_path = "/content/drive/MyDrive/botlab/VisDrone2019-MOT-val/annotations/uav0000339_00001_v.txt"

# Load image
img = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(img)

# Read annotations and filter frame 1
with open(ann_path, "r") as f:
    lines = f.readlines()

for line in lines:
    vals = line.strip().split(",")
    frame = int(vals[0])
    if frame == 1:
        _, track_id, x, y, w, h, _, cls, _, _ = map(int, vals)
        if cls == 1 or cls == 2:
          x1, y1 = x, y
          x2, y2 = x + w, y + h
          draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
          draw.text((x1, y1 - 10), f"ID {track_id}, C{cls}", fill="red")

# Show result
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis("off")
plt.show()