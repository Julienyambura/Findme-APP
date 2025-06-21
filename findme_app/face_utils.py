import face_recognition
import numpy as np
from fer import FER
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

emotion_detector = FER()

# Load OpenCV's pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces_and_emotions(image):
    face_locations = face_recognition.face_locations(image)
    emotions = []
    for (top, right, bottom, left) in face_locations:
        face = image[top:bottom, left:right]
        emotion, _ = emotion_detector.top_emotion(face)
        emotions.append(emotion if emotion else "Unknown")
    return face_locations, emotions

def draw_faces(image, locations, emotions):
    for i, (top, right, bottom, left) in enumerate(locations):
        color = (0, 255, 0)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        text = emotions[i] if i < len(emotions) else "Emotion?"
        cv2.putText(image, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def get_face_embedding(image):
    encodings = face_recognition.face_encodings(image)
    return encodings[0] if encodings else None

def is_similar_face(embedding, known_embedding, threshold=0.6):
    sim = cosine_similarity([embedding], [known_embedding])[0][0]
    return sim >= threshold

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image, len(faces)

st.title("👁️ FindMe+ – Face Detection (Streamlit)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    result_img, num_faces = detect_faces(image)
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption=f"Detected Faces: {num_faces}", use_column_width=True)
    st.success(f"Detected {num_faces} face(s) in the image.")
