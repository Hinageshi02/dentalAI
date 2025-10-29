import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import base64

# Flask setup
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("results", exist_ok=True)

# Fix for Ultralytics config directory warning
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# ✅ Load the YOLO model immediately on startup (safer for Render)
model_path = "runs/detect/cavity_yolo25/weights/best.pt"
model = YOLO(model_path)
print("✅ YOLOv8 cavity detection model loaded successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No file part"})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"})

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Run YOLOv8 inference
    results = model.predict(source=filepath, save=True, project="results", name="predictions", exist_ok=True)

    # Get annotated image
    result_path = results[0].save_dir / os.path.basename(filepath)
    annotated_img = cv2.imread(str(result_path))
    _, buffer = cv2.imencode(".jpg", annotated_img)
    result_base64 = base64.b64encode(buffer).decode("utf-8")

    # Confidence extraction (if available)
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        detections.append({"confidence": conf})

    return jsonify({
        "success": True,
        "message": "Detection completed successfully.",
        "result_image": f"data:image/jpeg;base64,{result_base64}",
        "detections": detections
    })


# ✅ Ensure Flask binds to the correct Render port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
