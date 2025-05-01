import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import numpy

#config
video_path = "13074060_1920_1080_30fps.mp4"
output_path = "named_output_1.mp4"
model = YOLO("yolov8n.pt")

#tracker
tracker = DeepSort(max_age=30)

id_map = {
    "person_1":"joey",
    "person_2": "Jim",
    "person_3":"cody",
    "person_4":"Bob",
    "person_5":"steve",
    "person_6":"josh",
    "sports ball_8":"basketball"
}

#video I/O setup
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),fps, (width,height))

frame_count=0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detection = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < 0.4:
            continue
        label = model.names[cls_id]
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
        detection.append(([x1,y1, x2-x1,y2-y1],conf, label))

    tracks = tracker.update_tracks(detection, frame=frame)

    for track in  tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l,t,r,b = track.to_ltrb()
        x1,y1,x2,y2 = int (l), int(t), int(r),int(b)
        track_label = track.get_det_class()

        raw_label = f"{track_label}_{track_id}"
        name_label = id_map.get(raw_label, raw_label)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, name_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
    out.write(frame)
    frame_count += 1


cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[done] Saved annotated video as: {output_path}")