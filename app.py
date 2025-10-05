import os
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from ultralytics import YOLO
import cv2

# ---------------------------
# 🔧 Configuration
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 👇 Change ONLY this path if your best.pt is in a different folder
MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs", "detect", "cavity_yolo25", "weights", "best.pt"
)

# ---------------------------
# 🚨 Check model existence
# ---------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

# ---------------------------
# 🚀 Initialize Flask app
# ---------------------------
app = Flask(__name__, static_folder="results", template_folder="templates")

# Load YOLOv8 model
model = YOLO(MODEL_PATH)
print("✅ YOLOv8 cavity detection model loaded successfully!")

# ---------------------------
# 🏠 Home Route
# ---------------------------
@app.route("/")
def home():
    """Render the upload web page"""
    return render_template("index.html")

# ---------------------------
# 🧠 Prediction Endpoint
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    result_path = os.path.join(RESULT_FOLDER, f"detected_{file.filename}")
    file.save(image_path)

    try:
        # -------------------------------
        # 🔹 Preprocess: upscale small images
        # -------------------------------
        img = cv2.imread(image_path)
        height, width = img.shape[:2]

        MIN_HEIGHT = 416
        MIN_WIDTH = 416

        if height < MIN_HEIGHT or width < MIN_WIDTH:
            scale = max(MIN_HEIGHT / height, MIN_WIDTH / width)
            new_w, new_h = int(width * scale), int(height * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            temp_path = os.path.join(UPLOAD_FOLDER, f"resized_{file.filename}")
            cv2.imwrite(temp_path, img)
            inference_path = temp_path
        else:
            inference_path = image_path

        # -------------------------------
        # 🔹 Run YOLOv8 inference
        # -------------------------------
        results = model(inference_path, imgsz=640, conf=0.4)

        # -------------------------------
        # 🔹 Draw detections with confidence-based colors
        # -------------------------------
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)

                # Set color based on confidence
                if conf >= 0.5:
                    color = (0, 0, 255)      # Red = high confidence
                elif conf >= 0.3:
                    color = (0, 165, 255)    # Orange = medium
                else:
                    color = (0, 255, 255)    # Yellow = low

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, f"Cavity {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf)
                })

        # Save the final result
        cv2.imwrite(result_path, img)

        # Return a URL for the frontend
        result_url = url_for("serve_result_image", filename=f"detected_{file.filename}")
        return jsonify({
            "success": True,
            "message": "Cavity detection completed.",
            "result_image": result_url,
            "detections": detections
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"success": False, "error": str(e)})

# ---------------------------
# 🖼 Serve processed images
# ---------------------------
@app.route("/results/<path:filename>")
def serve_result_image(filename):
    """Serve the resulting image for display in browser"""
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
