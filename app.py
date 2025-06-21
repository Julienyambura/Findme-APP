# findme_app/app.py
import streamlit as st
import cv2
import numpy as np
import os
import json
import logging
from datetime import datetime
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load OpenCV's pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known face embeddings - updated path for root directory
DB_PATH = os.path.join("database", "data.json")

# Ensure database directory exists
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image, len(faces)

def add_face_to_database(name, image):
    """Add a face to the database"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            # Save the face data to the database
            data = {}
            if os.path.exists(DB_PATH):
                with open(DB_PATH, 'r') as f:
                    data = json.load(f)
            
            # Convert image to base64 for storage
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            data[name] = {
                'image': img_base64,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(DB_PATH, 'w') as f:
                json.dump(data, f)
                
            return True, "Face detected and added to database"
        else:
            return False, "No face detected in the image"
    except Exception as e:
        logger.error(f"Error processing image for database: {str(e)}")
        return False, f"Error processing image: {str(e)}"

# Streamlit UI
st.title("👁️ FindMe+ – Face Detection (Streamlit)")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Face Detection", "Add Missing Person"])

if page == "Face Detection":
    st.header("Face Detection")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        result_img, num_faces = detect_faces(image)
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption=f"Detected Faces: {num_faces}", use_column_width=True)
        st.success(f"Detected {num_faces} face(s) in the image.")

elif page == "Add Missing Person":
    st.header("Add Missing Person to Database")
    
    name = st.text_input("Enter the person's name:")
    uploaded_file = st.file_uploader("Upload an image of the person", type=["jpg", "jpeg", "png"])
    
    if st.button("Add to Database") and name and uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        success, message = add_face_to_database(name, image)
        
        if success:
            st.success(message)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Added: {name}", use_column_width=True)
        else:
            st.error(message)
