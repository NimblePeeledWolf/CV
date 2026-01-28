from ultralytics import YOLO
import cv2
from PIL import Image
import imageio.v3 as iio
import os


model= YOLO("yolov8n.pt")

results = model(["/workspaces/CV/practice nike"])

for i, result in enumerate(results):
    annotated_frame = result.plot()
    output_path = f"/workspaces/CV/outputs/detected_image{i+1}.jpeg"
    cv2.imwrite(output_path, annotated_frame)

