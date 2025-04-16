from ultralytics import YOLO
import cv2
from PIL import Image
import imageio.v3 as iio
import os

avif_files = [
    "/workspaces/CV/static/photo-1741606369311-8e3d4dbe4b8e.avif"
    "/workspaces/CV/static/premium_photo-1676977396095-07e0648d92df.avif"
    "static/premium_photo-1742404281241-79bbcd1f8dab.avif"
]

output_folder = "static"

for avif_path in avif_files:
    #load in avif images
    img = iio.imread(avif_path)
    #convert to pil image and define output path
    img_pil = Image.fromarray(img)
    base_name = os.path.splitext(os.path.basename(avif_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}.jpg")

    #save as JPEG
    img_pil.save(output_path, "JPEG")


model = YOLO("yolov8n.pt")
img_path = "/workspaces/CV/static/thumbnail_image0.jpg"
results = model(img_path)

annotated_frame = results[0].plot()

output_path = "/workspaces/CV/outputs/detected_sample_2.jpg"
cv2.imwrite(output_path, annotated_frame)

print(f"Detection Complete. Output saved to: {output_path}")