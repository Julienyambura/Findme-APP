import streamlit as st
import cv2
import numpy as np

st.title("👁️ FindMe+ – Face Detection")

# Load OpenCV's pre-trained Haar Cascade face detector
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    st.success("✅ Face detection model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading face detection model: {e}")
    st.stop()

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

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Convert uploaded file to image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is not None:
            # Detect faces
            result_img, num_faces = detect_faces(image)
            
            # Display results
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

# Add some helpful information
st.markdown("---")
st.markdown("### How to use:")
st.markdown("1. Upload an image containing faces")
st.markdown("2. The app will automatically detect and highlight faces with green rectangles")
st.markdown("3. You'll see the count of detected faces")

# Test the environment
st.markdown("---")
st.markdown("### Environment Check:")
st.markdown(f"✅ Streamlit version: {st.__version__}")
st.markdown(f"✅ OpenCV version: {cv2.__version__}")
st.markdown(f"✅ NumPy version: {np.__version__}")
