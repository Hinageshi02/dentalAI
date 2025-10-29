import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import base64

# ---------------------------
# 🔧 Flask App Configuration
# ---------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

# Ensure directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("results", exist_ok=True)

# Avoid permission issues in Render
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# ---------------------------
# 🚀 Load YOLO Model (once)
# ---------------------------
MODEL_PATH = "runs/detect/cavity_yolo25/weights/best.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

model = YOLO(MODEL_PATH)
print("✅ YOLOv8 cavity detection model loaded successfully!")


# ---------------------------
# 🏠 Home Page
# ---------------------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------
# 🧠 Prediction Endpoint
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Run YOLO prediction
    results = model.predict(source=filepath, save=True, project="results", name="predictions", exist_ok=True)

    # Get the annotated image path
    result_path = results[0].save_dir / os.path.basename(filepath)
    annotated = cv2.imread(str(result_path))

    # Convert to base64 for frontend display
    _, buffer = cv2.imencode(".jpg", annotated)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    # Extract detections
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        detections.append({"confidence": conf})

    return jsonify({
        "success": True,
        "message": "Cavity detection completed successfully.",
        "result_image": f"data:image/jpeg;base64,{encoded_image}",
        "detections": detections
    })


# ---------------------------
# 🧩 Render Entry Point
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
