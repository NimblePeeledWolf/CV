from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import uuid
import cv2


app = Flask(__name__)

model_path = os.environ.get('MODEL_PATH', 'best.pt')
upload_dir = "static/uploads"
output_dir = "static/outputs"

os.makedirs(upload_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_path)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["image"]
    if file.filename == "":
        return "Empty filename", 400
    
    uid = uuid.uuid4.hex
    in_path = os.path.join(upload_dir, f"{uid}jpg")
    out_path = os.path.join(output_dir, f"{uid}jpg")

    file.save(in_path)

    result = model(in_path, imgsz = 640)[0]
    annotated = result.plot() 
    cv2.imwrite(out_path, annotated)

    #extract detection summary
    detections = []
    names = result.names
    if result.boxes is not None and len(result.boxes) > 0:
        for b in result.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            detections.append({"label": names[cls_id], "conf": round(conf, 3)})

    return render_template("result.html", input_image=in_path, output_image=out_path, detections=detections)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)