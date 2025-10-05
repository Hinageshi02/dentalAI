import os
import cv2
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from ultralytics import YOLO

# ---------------------------
# 🔧 Configuration
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "runs", "detect", "cavity_yolo25", "weights")
MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 👇 CHANGE THIS LINK to your own Google Drive direct download link
MODEL_URL = "https://drive.google.com/uc?id=17sVcw1WvqzrU3vVNpjKmdoyrjDIj9hAI"

# ---------------------------
# 📦 Download model if not found
# ---------------------------
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading YOLO model from Google Drive...")
    r = requests.get(MODEL_URL, allow_redirects=True)
    if r.status_code == 200:
        open(MODEL_PATH, 'wb').write(r.content)
        print("✅ Model downloaded successfully!")
    else:
        raise Exception(f"❌ Failed to download model. Status code: {r.status_code}")

# ---------------------------
# 🚨 Verify model file
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
        # Run YOLOv8 inference
        results = model(image_path, conf=0.4)

        # Draw detections using OpenCV
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

        # ✅ Return URL for the detected image
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
# 🖼 Serve processed images
# ---------------------------
@app.route("/results/<path:filename>")
def serve_result_image(filename):
    """Serve the resulting image for display in browser"""
    return send_from_directory(RESULT_FOLDER, filename)

# ---------------------------
# 🏁 Run Flask App
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render uses dynamic port
    app.run(host="0.0.0.0", port=port, debug=False)
