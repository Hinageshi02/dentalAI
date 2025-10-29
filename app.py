import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image, ImageDraw
import io
import random
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# ---------- Homepage ----------
@app.route('/')
def index():
    return render_template('index.html')

# ---------- Prediction Route ----------
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Open image for mock detection (replace this with your AI model if available)
        img = Image.open(filepath).convert("RGB")
        draw = ImageDraw.Draw(img)

        detections = []
        # Simulate 0–3 random "cavity detections"
        for _ in range(random.randint(0, 3)):
            x1, y1 = random.randint(20, 150), random.randint(20, 150)
            x2, y2 = x1 + random.randint(50, 100), y1 + random.randint(50, 100)
            conf = round(random.uniform(0.2, 0.9), 2)
            detections.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'confidence': conf})

            color = "red" if conf >= 0.5 else "orange" if conf >= 0.3 else "yellow"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Save annotated image
        result_filename = f"result_{file.filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        img.save(result_path)

        return jsonify({
            'success': True,
            'result_image': f"/{result_path}",
            'detections': detections
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ---------- Serve Static Files ----------
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# ---------- Run Local (ignored by Render) ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
