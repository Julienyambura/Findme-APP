import face_recognition
import numpy as np
from fer import FER
import cv2
from sklearn.metrics.pairwise import cosine_similarity

emotion_detector = FER()

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
