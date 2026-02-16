from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
import os
import uuid
import cv2


app = Flask(__name__)
MODEL_PATH = "/workspaces/CV/runs/detect/train4/weights/best.pt"
model_path = os.environ.get("MODEL_PATH", MODEL_PATH)
upload_dir = os.path.join(app.root_path, "static", "uploads")
output_dir = os.path.join(app.root_path, "static", "outputs")

os.makedirs(upload_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

model = YOLO(model_path)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file or file.filename == "":
        return "No file uploaded", 400

    uid = uuid.uuid4().hex
    in_filename = f"{uid}.jpg"
    out_filename = f"{uid}_det.jpg"

    in_path = os.path.join(upload_dir, in_filename)
    out_path = os.path.join(output_dir, out_filename)

    file.save(in_path)

    result = model(in_path, imgsz=640, iou=0.20)[0]
    annotated = result.plot()
    cv2.imwrite(out_path, annotated)

    detections = []
    names = result.names
    if result.boxes is not None and len(result.boxes) > 0:
        for b in result.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            detections.append({"label": names[cls_id], "conf": round(conf, 3)})

    return render_template(
        "result.html",
        input_image=url_for("static", filename=f"uploads/{in_filename}"),
        output_image=url_for("static", filename=f"outputs/{out_filename}"),
        detections=detections
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)