
from flask import Flask, jsonify, request
import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
import os
import tempfile
import werkzeug
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for face detection
@app.route('/detect_face', methods=['POST'])
def detect_face():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        return jsonify({"face_locations": face_locations}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for anti-spoofing (placeholder logic)
@app.route('/anti_spoofing', methods=['POST'])
def anti_spoofing():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        # Placeholder for anti-spoofing logic
        # Implement actual anti-spoofing detection logic here
        result = "Spoofing detection logic is not implemented."
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for deepfake detection
@app.route('/deepfake_detection', methods=['POST'])
def deepfake_detection():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        # Use a temporary file to handle uploads
        with tempfile.NamedTemporaryFile(delete=False, suffix=werkzeug.utils.secure_filename(file.filename)) as temp_file:
            file.save(temp_file.name)
            # Use DeepFace for deepfake detection
            result = DeepFace.analyze(temp_file.name, actions=['deepfake'])

        os.remove(temp_file.name)  # Clean up the temporary file
        return jsonify({"result": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)  # Set debug=False for production
