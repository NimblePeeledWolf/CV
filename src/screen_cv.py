import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import numpy


model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("13074060_1920_1080_30fps.mp4")

tracker = DeepSort(max_age=30)

os.makedirs("tracked", exist_ok=True)

frame_count=0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results= model(frame)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.4:
            continue
        label = model.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        detections.append(([x1 , y1, x2 - x1, y2 - y1], conf, label))
    
    tracks = tracker.update_tracks(detections,frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        x1, y1, x2, y2 = int(l), int(t), int(w), int(h)
        track_label= track.get_det_class()

        folder_name = f"{track_label}_{track_id}"
        crop__dir = os.path.join("tracked", folder_name)
        os.makedirs(crop__dir,exist_ok=True)

        crop = frame[y1:y2, x1:x2]
        crop_path = os.path.join(crop__dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(crop_path, crop)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
