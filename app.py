import os
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import cv2
import base64

# -----------------------------
# 🔧 Flask Configuration
# -----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('results', exist_ok=True)

# -----------------------------
# ⚙️ Model Path Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "runs", "detect", "cavity_yolo25", "weights", "best.pt"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

# Load YOLO model
model = YOLO(MODEL_PATH)
print("✅ YOLO cavity detection model loaded successfully!")

# -----------------------------
# 🏠 Home Route
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

# -----------------------------
# 🧠 Prediction Route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Run YOLO prediction
    results = model.predict(source=filepath, save=True, project='results', name='predictions', exist_ok=True)

    # Get annotated image
    result_path = results[0].save_dir / os.path.basename(filepath)
    annotated = cv2.imread(str(result_path))

    # Convert to base64
    _, buffer = cv2.imencode('.jpg', annotated)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    # Extract detections
    detections = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        detections.append({'confidence': conf})

    return jsonify({
        'success': True,
        'message': 'Cavity detection completed successfully.',
        'result_image': f"data:image/jpeg;base64,{encoded_image}",
        'detections': detections
    })

# -----------------------------
# 🚀 Local Run (Ignored by Render)
# -----------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
