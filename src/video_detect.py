import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture("13074060_1920_1080_30fps.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('annotated_vdeo.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width , height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame =results[0].plot()

    out.write(annotated_frame)

cap.release()
out.release()
cv2.destroyAllWindows()