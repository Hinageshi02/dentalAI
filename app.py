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

# 👇 YOLO model path (change only this if needed)
MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs", "detect", "cavity_yolo25", "weights", "best.pt"
)

# ---------------------------
# 🚨 Check model existence
# ---------------------------
if not os.path.exists(MODEL_PATH):
    print(f"⚠️ Model not found locally at {MODEL_PATH}")
    print("➡️ You can download it manually or host it on cloud storage (Google Drive, etc.)")
else:
    print(f"✅ Found model at: {MODEL_PATH}")

# ---------------------------
# 🚀 Initialize Flask app
# ---------------------------
app = Flask(__name__, static_folder="results", template_folder="templates")

# ---------------------------
# 🧠 Load YOLO model
# ---------------------------
try:
    model = YOLO(MODEL_PATH)
    print("✅ YOLOv8 cavity detection model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    model = None


# ---------------------------
# 🏠 Home Route
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
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    result_path = os.path.join(RESULT_FOLDER, f"detected_{file.filename}")
    file.save(image_path)

    if model is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500

    try:
        results = model(image_path, conf=0.4)
        img = cv2.imread(image_path)

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img, f"Cavity {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imwrite(result_path, img)
        result_url = url_for("serve_result_image", filename=f"detected_{file.filename}")

        return jsonify({
            "success": True,
            "message": "Cavity detection completed.",
            "result_image": result_url
        })

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"success": False, "error": str(e)})


# ---------------------------
# 🖼 Serve Result Images
# ---------------------------
@app.route("/results/<path:filename>")
def serve_result_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)


# ---------------------------
# 🏁 Run Flask App (Render-compatible)
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # ✅ Render sets PORT automatically
    app.run(host="0.0.0.0", port=port, debug=True)
