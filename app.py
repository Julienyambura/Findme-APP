import streamlit as st
import cv2
import numpy as np
import os
import json
import base64
from datetime import datetime
from PIL import Image

st.title("👁️ FindMe+ – Face Detection & Recognition")

# Load OpenCV's pre-trained Haar Cascade face detector
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    st.success("✅ Face detection model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading face detection model: {e}")
    st.stop()

# Database setup
DB_PATH = "database/data.json"
os.makedirs("database", exist_ok=True)

def load_database():
    """Load the face database"""
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_database(data):
    """Save the face database"""
    with open(DB_PATH, 'w') as f:
        json.dump(data, f, indent=2)

def extract_face_features(image):
    """Extract face features using simple histogram comparison"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        gray = cv2.resize(gray, (64, 64))
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        
        # Normalize histogram
        hist = cv2.normalize(hist, hist).flatten()
        
        return hist
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def compare_faces(features1, features2):
    """Compare two face feature vectors using correlation"""
    try:
        if features1 is None or features2 is None:
            return 0.0
        
        # Calculate correlation coefficient
        correlation = cv2.compareHist(features1, features2, cv2.HISTCMP_CORREL)
        
        # Convert to similarity score (0-1)
        similarity = (correlation + 1) / 2
        return max(0, similarity)
    except Exception as e:
        st.error(f"Error comparing faces: {e}")
        return 0.0

def detect_faces(image):
    """Detect faces in an image"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return image, len(faces)
    except Exception as e:
        st.error(f"Error detecting faces: {e}")
        return image, 0

def recognize_face(uploaded_image, database):
    """Recognize faces in uploaded image against database"""
    try:
        # Convert uploaded image to numpy array
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return None, "Could not read the uploaded image"
        
        # Get face features for uploaded image
        uploaded_features = extract_face_features(image)
        if uploaded_features is None:
            return None, "No face detected in uploaded image"
        
        # Compare with database
        matches = []
        for name, data in database.items():
            try:
                # Decode stored image
                img_data = base64.b64decode(data['image'])
                stored_image = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                
                # Get features for stored image
                stored_features = extract_face_features(stored_image)
                if stored_features is not None:
                    # Compare faces
                    similarity = compare_faces(uploaded_features, stored_features)
                    
                    if similarity > 0.8:  # Threshold for matching
                        matches.append({
                            'name': name,
                            'similarity': similarity,
                            'timestamp': data.get('timestamp', 'Unknown')
                        })
            except Exception as e:
                st.warning(f"Error processing {name}: {e}")
                continue
        
        return matches, None
    except Exception as e:
        return None, f"Error during recognition: {e}"

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["Face Detection", "Add Person to Database", "Face Recognition"])

if page == "Face Detection":
    st.header("Face Detection")
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is not None:
                result_img, num_faces = detect_faces(image)
                st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), 
                        caption=f"Detected {num_faces} face(s)", 
                        use_column_width=True)
                
                if num_faces > 0:
                    st.success(f"✅ Found {num_faces} face(s) in the image!")
                else:
                    st.warning("⚠️ No faces detected in the image.")
            else:
                st.error("❌ Could not read the uploaded image.")
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

elif page == "Add Person to Database":
    st.header("Add Person to Database")
    
    name = st.text_input("Enter the person's name:")
    uploaded_file = st.file_uploader("Upload an image of the person", type=["jpg", "jpeg", "png"])
    
    if st.button("Add to Database") and name and uploaded_file:
        try:
            # Convert uploaded file to image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is not None:
                # Check if face is detected
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Extract face features
                    features = extract_face_features(image)
                    if features is not None:
                        # Load database
                        database = load_database()
                        
                        # Convert image to base64 for storage
                        _, buffer = cv2.imencode('.jpg', image)
                        img_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Add to database
                        database[name] = {
                            'image': img_base64,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        save_database(database)
                        st.success(f"✅ {name} added to database successfully!")
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Added: {name}", use_column_width=True)
                    else:
                        st.error("❌ Could not extract face features. Please try a different image.")
                else:
                    st.error("❌ No face detected in the image. Please upload an image with a clear face.")
            else:
                st.error("❌ Could not read the uploaded image.")
        except Exception as e:
            st.error(f"❌ Error adding to database: {e}")

elif page == "Face Recognition":
    st.header("Face Recognition")
    
    # Show database stats
    database = load_database()
    st.info(f"📊 Database contains {len(database)} person(s)")
    
    if len(database) > 0:
        st.subheader("People in Database:")
        for name, data in database.items():
            st.write(f"• {name} (added: {data.get('timestamp', 'Unknown')[:10]})")
    
    uploaded_file = st.file_uploader("Upload an image to recognize", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and st.button("Recognize Face"):
        matches, error = recognize_face(uploaded_file, database)
        
        if error:
            st.error(f"❌ {error}")
        elif matches:
            st.success("🎯 Face Recognition Results:")
            for match in sorted(matches, key=lambda x: x['similarity'], reverse=True):
                similarity_percent = match['similarity'] * 100
                st.write(f"✅ **{match['name']}** - {similarity_percent:.1f}% match")
                st.write(f"   Added: {match['timestamp'][:10]}")
        else:
            st.warning("⚠️ No matches found in database")

# Add some helpful information
st.markdown("---")
st.markdown("### How to use:")
st.markdown("1. **Face Detection**: Upload any image to detect faces")
st.markdown("2. **Add Person**: Add faces to the database with names")
st.markdown("3. **Face Recognition**: Upload an image to find matches in the database")

# Test the environment
st.markdown("---")
st.markdown("### Environment Check:")
st.markdown(f"✅ Streamlit version: {st.__version__}")
st.markdown(f"✅ OpenCV version: {cv2.__version__}")
st.markdown(f"✅ NumPy version: {np.__version__}")
try:
    import PIL
    st.markdown(f"✅ Pillow version: {PIL.__version__}")
except:
    st.error("❌ Pillow not available")
