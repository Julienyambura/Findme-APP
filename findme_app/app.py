# findme_app/app.py
import streamlit as st
import cv2
import numpy as np
import os
import json
from mtcnn import MTCNN
from fer import FER
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Initialize MTCNN and FER
    detector = MTCNN()
    emotion_detector = FER(mtcnn=True)
except Exception as e:
    logger.error(f"Error initializing detectors: {str(e)}")
    st.error("Error initializing face detection. Please try again later.")
    st.stop()

# Load known face embeddings
DB_PATH = "database/data.json"

def detect_faces_and_emotions(image):
    try:
        # Detect faces using MTCNN
        faces = detector.detect_faces(image)
        face_locations = []
        emotions = []
        
        for face in faces:
            x, y, w, h = face['box']
            face_locations.append((y, x + w, y + h, x))
            
            # Get emotion for the face
            face_img = image[y:y+h, x:x+w]
            if face_img.size > 0:
                try:
                    emotion = emotion_detector.detect_emotions(face_img)
                    if emotion:
                        emotions.append(max(emotion[0]['emotions'].items(), key=lambda x: x[1])[0])
                    else:
                        emotions.append("Unknown")
                except Exception as e:
                    logger.error(f"Error detecting emotion: {str(e)}")
                    emotions.append("Unknown")
        
        return face_locations, emotions
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        return [], []

def draw_faces(image, face_locations, emotions):
    try:
        for (top, right, bottom, left), emotion in zip(face_locations, emotions):
            # Draw rectangle around face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add emotion text
            cv2.putText(image, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image
    except Exception as e:
        logger.error(f"Error drawing faces: {str(e)}")
        return image

st.set_page_config(page_title="FindMe+", layout="centered")
st.title("👁️ FindMe+ – Face Detection + Emotion AI")

menu = st.sidebar.selectbox("Choose Action", ["Live Camera", "Detect & Match", "Add Missing Person"])

if menu == "Live Camera":
    st.subheader("📷 Live Webcam Detection")
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam. Please make sure your webcam is connected and try again.")
            st.stop()
    except Exception as e:
        logger.error(f"Error accessing webcam: {str(e)}")
        st.error("Error accessing webcam. Please try again later.")
        st.stop()

    while run:
        try:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations, emotions = detect_faces_and_emotions(frame_rgb)
            frame_annotated = draw_faces(frame_rgb.copy(), face_locations, emotions)

            FRAME_WINDOW.image(frame_annotated)
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            st.error("Error processing video frame. Please try again.")
            break
    cap.release()

elif menu == "Detect & Match":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations, emotions = detect_faces_and_emotions(frame_rgb)
            frame_with_faces = draw_faces(frame_rgb.copy(), face_locations, emotions)

            st.image(frame_with_faces, caption="Detected Faces & Emotions", use_column_width=True)

            if face_locations:
                for emotion in emotions:
                    st.info(f"Detected Emotion: {emotion}")
            else:
                st.warning("No faces detected.")
        except Exception as e:
            logger.error(f"Error processing uploaded image: {str(e)}")
            st.error("Error processing image. Please try again with a different image.")

elif menu == "Add Missing Person":
    st.subheader("📤 Add to Missing Person Database")
    name = st.text_input("Enter Name")
    add_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if name and add_file:
        try:
            file_bytes = np.asarray(bytearray(add_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            faces = detector.detect_faces(img_rgb)
            if faces:
                st.success("Face detected and added to database.")
            else:
                st.error("No face detected in the image.")
        except Exception as e:
            logger.error(f"Error processing image for database: {str(e)}")
            st.error("Error processing image. Please try again with a different image.")
