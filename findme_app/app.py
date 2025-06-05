# findme_app/app.py
import streamlit as st
import cv2
import numpy as np
import os
import json
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Remove this line
from face_utils import detect_faces_and_emotions, draw_faces, get_face_embedding, is_similar_face

# Load known face embeddings
DB_PATH = "database/data.json"

def load_known_embeddings():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f:
            return json.load(f)
    return []

def save_new_person(name, embedding):
    db = load_known_embeddings()
    db.append({"name": name, "embedding": embedding})
    with open(DB_PATH, 'w') as f:
        json.dump(db, f)

st.set_page_config(page_title="FindMe+", layout="centered")
st.title("👁️ FindMe+ – Face Recognition + Emotion AI")

menu = st.sidebar.selectbox("Choose Action", ["Live Camera", "Detect & Match", "Add Missing Person"])

if menu == "Live Camera":
    st.subheader("📷 Live Webcam Detection")
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations, emotions = detect_faces_and_emotions(frame_rgb)
        frame_annotated = draw_faces(frame_rgb.copy(), face_locations, emotions)

        # Face matching
        db = load_known_embeddings()
        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_img = frame_rgb[top:bottom, left:right]
            embedding = get_face_embedding(face_img)
            if embedding is not None:
                for person in db:
                    if is_similar_face(embedding, person['embedding']):
                        cv2.putText(frame_annotated, f"Match: {person['name']}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        break

        FRAME_WINDOW.image(frame_annotated)
    cap.release()

elif menu == "Detect & Match":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations, emotions = detect_faces_and_emotions(frame_rgb)
        frame_with_faces = draw_faces(frame_rgb.copy(), face_locations, emotions)

        st.image(frame_with_faces, caption="Detected Faces & Emotions", use_column_width=True)

        if face_locations:
            db = load_known_embeddings()
            for i, (top, right, bottom, left) in enumerate(face_locations):
                face_img = frame_rgb[top:bottom, left:right]
                embedding = get_face_embedding(face_img)

                if embedding is not None:
                    for person in db:
                        if is_similar_face(embedding, person['embedding']):
                            st.success(f"Match: {person['name']} - Emotion: {emotions[i]}")
                            break
                    else:
                        st.info(f"No match found - Emotion: {emotions[i]}")
                else:
                    st.warning("Could not extract face encoding.")
        else:
            st.warning("No faces detected.")

elif menu == "Add Missing Person":
    st.subheader("📤 Add to Missing Person Database")
    name = st.text_input("Enter Name")
    add_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if name and add_file:
        file_bytes = np.asarray(bytearray(add_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        embedding = get_face_embedding(img_rgb)

        if embedding is not None:
            save_new_person(name, embedding.tolist())
            st.success("Person added to database.")
        else:
            st.error("Face not clear or detectable.")
