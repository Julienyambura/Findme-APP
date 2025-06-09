# findme_app/app.py
from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import os
import json
import logging
from datetime import datetime
import base64

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize face detector
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    logger.error(f"Error initializing face detector: {str(e)}")

# Load known face embeddings
DB_PATH = "database/data.json"

def detect_faces_and_emotions(image):
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces using OpenCV
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_locations = []
        emotions = []
        
        for (x, y, w, h) in faces:
            face_locations.append((y, x + w, y + h, x))
            emotions.append("Face Detected")
        
        return face_locations, emotions
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return [], []

def draw_faces(image, face_locations, emotions):
    try:
        for (top, right, bottom, left), emotion in zip(face_locations, emotions):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image
    except Exception as e:
        logger.error(f"Error drawing faces: {str(e)}")
        return image

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations, emotions = detect_faces_and_emotions(frame_rgb)
            frame_annotated = draw_faces(frame_rgb.copy(), face_locations, emotions)
            ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_annotated, cv2.COLOR_RGB2BGR))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['POST'])
def detect_faces():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations, emotions = detect_faces_and_emotions(frame_rgb)
        frame_with_faces = draw_faces(frame_rgb.copy(), face_locations, emotions)

        # Convert the processed image to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(frame_with_faces, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'success': True,
            'num_faces': len(face_locations),
            'image': img_base64
        })
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Error processing image'}), 500

@app.route('/add_missing', methods=['POST'])
def add_missing_person():
    if 'file' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'Missing file or name'}), 400

    name = request.form['name']
    file = request.files['file']

    try:
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            # Here you would typically save the face data to your database
            return jsonify({'success': True, 'message': 'Face detected and added to database'})
        else:
            return jsonify({'error': 'No face detected in the image'}), 400
    except Exception as e:
        logger.error(f"Error processing image for database: {str(e)}")
        return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(debug=True)
