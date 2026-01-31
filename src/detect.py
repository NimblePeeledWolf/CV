from ultralytics import YOLO
import cv2
from PIL import Image
import imageio.v3 as iio
import os



model = YOLO("yolov8n.pt")
img_path = "//workspaces/CV/nike/symbol.n.jpeg"
results = model(img_path)

annotated_frame = results[0].plot()

output_path = "/workspaces/CV/outputs/detected_sample.jpg"
cv2.imwrite(output_path, annotated_frame)

print(f"Detection Complete. Output saved to: {output_path}")