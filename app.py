import os
import torch
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import base64

# ======================================================
# ⚙️ Flask App Config
# ======================================================
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("results", exist_ok=True)

# ======================================================
# 🧠 Safe YOLO Model Loading (Torch 2.6+)
# ======================================================
MODEL_PATH = os.path.join("runs", "detect", "cavity_yolo25", "weights", "best.pt")
model = None

try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

    model = YOLO(MODEL_PATH)
    print("✅ YOLO model loaded successfully!")

except Exception as e:
    print(f"⚠️ Failed to load YOLO model: {e}")
    print("Running in mock mode (no real detection).")

# ======================================================
# 🏠 Home Route
# ======================================================
@app.route("/")
def home():
    return render_template("index.html")

# ======================================================
# 🧩 Predict Route
# ======================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        if model is None:
            # ---- Mock fallback (for low-memory Render) ----
            img = cv2.imread(filepath)
            h, w, _ = img.shape
            cv2.rectangle(img, (20, 20), (w - 20, h - 20), (0, 0, 255), 2)
            _, buffer = cv2.imencode(".jpg", img)
            encoded_image = base64.b64encode(buffer).decode("utf-8")
            return jsonify({
                "success": True,
                "message": "Mock detection mode (no model loaded).",
                "result_image": f"data:image/jpeg;base64,{encoded_image}",
                "detections": []
            })

        # ---- Real YOLO prediction ----
        results = model.predict(
            source=filepath,
            imgsz=320,
            device="cpu",
            conf=0.25,
            save=True,
            project="results",
            name="predictions",
            exist_ok=True
        )

        result_path = results[0].save_dir / os.path.basename(filepath)
        annotated = cv2.imread(str(result_path))

        _, buffer = cv2.imencode(".jpg", annotated)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        detections = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            detections.append({"confidence": conf})

        return jsonify({
            "success": True,
            "message": "Detection completed successfully.",
            "result_image": f"data:image/jpeg;base64,{encoded_image}",
            "detections": detections
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ======================================================
# 🚀 Render Entry Point
# ======================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
