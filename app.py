import os
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
import cv2
from ultralytics import YOLO

# ---------------------------
# 🔧 Configuration
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "cavity_yolo25", "weights", "best.pt")

app = Flask(__name__, static_folder="results", template_folder="templates")

# 🧠 Lazy model loading
model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global model

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    file = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    result_path = os.path.join(RESULT_FOLDER, f"detected_{file.filename}")
    file.save(image_path)

    try:
        # ✅ Load model lazily (only once)
        if model is None:
            if not os.path.exists(MODEL_PATH):
                return jsonify({"success": False, "error": f"Model not found at {MODEL_PATH}"}), 500
            print("🧠 Loading YOLO model for the first time (CPU mode)...")
            model = YOLO(MODEL_PATH)
            model.to("cpu")  # ✅ Force CPU mode for Render
            print("✅ Model loaded successfully!")

        # ✅ Run with minimal memory
        results = model.predict(image_path, conf=0.45, verbose=False, device="cpu", imgsz=320)

        img = cv2.imread(image_path)
        if img is None:
            return jsonify({"success": False, "error": "Failed to read uploaded image."}), 500

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"Cavity {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

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


@app.route("/results/<path:filename>")
def serve_result_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
