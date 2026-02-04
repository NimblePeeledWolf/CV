from ultralytics import YOLO
import cv2
from glob import glob
import os



model = YOLO("yolov8n.pt")
input_root = "/workspaces/CV/dataset_raw"
output_root = "/workspaces/CV/detected"
os.makedirs(output_root,exist_ok=True)

#collect images from folders
image_paths = []
for brand_folder in os.listdir(input_root):
    full_path = os.path.join(input_root, brand_folder)
    if os.path.isdir(full_path):
        for ext in ("*.jpeg","*.jpg","*.png","*.avif","*.webp"):
            image_paths.extend(glob(os.path.join(full_path, ext)))

if not image_paths:
    raise FileNotFoundError(f"No images found under {input_root}")

for img_path in image_paths:
    # run inference on ONE image (prevents memory blowups)
    result = model(img_path)[0]
    annotated = result.plot()

    brand = os.path.basename(os.path.dirname(img_path))
    out_dir = os.path.join(output_root, brand)
    os.makedirs(out_dir, exist_ok=True)

    filename = os.path.splitext(os.path.basename(img_path))[0] + "_det.jpg"
    out_path = os.path.join(out_dir, filename)

    cv2.imwrite(out_path, annotated)

print(f"done. saved {len(image_paths)} images to {output_root}")