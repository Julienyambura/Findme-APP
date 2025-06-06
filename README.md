# FindMe+ - Face Recognition & Emotion AI App

A Streamlit application that combines face recognition and emotion detection capabilities.

## Features

- Live webcam face detection and emotion recognition
- Upload images for face detection and matching
- Add missing persons to the database
- Real-time emotion analysis

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run findme_app/app.py
```

## Requirements

- Python 3.10.13
- See requirements.txt for full list of dependencies

## Project Structure

- `findme_app/app.py`: Main Streamlit application
- `findme_app/face_utils.py`: Face detection and recognition utilities
- `database/`: Storage for face embeddings
