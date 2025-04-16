from ultralytics import YOLO
import cv2
from PIL import Image

model = YOLO("yolov8n.pt")
img_path = "/workspaces/CV/static/thumbnail_image0.jpg"
results = model(img_path)

annotated_frame = results[0].plot()

output_path = "/workspaces/CV/outputs/detected_sample_2.jpg"
cv2.imwrite(output_path, annotated_frame)

print(f"Detection Complete. Output saved to: {output_path}")