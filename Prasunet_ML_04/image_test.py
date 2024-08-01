import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the trained model
model = load_model('handG_Rec.h5')

# Class labels
class_labels = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']

# Gesture to operation mapping
gesture_operations = {
    'palm': 'Operation 1: Open Hand',
    'l': 'Operation 2: L Shape',
    'fist': 'Operation 3: Closed Fist',
    'fist_moved': 'Operation 4: Moving Fist',
    'thumb': 'Operation 5: Thumb Up',
    'index': 'Operation 6: Pointing with Index',
    'ok': 'Operation 7: OK Sign',
    'palm_moved': 'Operation 8: Moving Open Hand',
    'c': 'Operation 9: C Shape',
    'down': 'Operation 10: Down Gesture'
}

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (150, 150))  # Resize to the expected input size of the model
    image = image / 255.0  # Normalize the image
    image = img_to_array(image)
    return np.expand_dims(image, axis=0)

# Streamlit UI
st.title("Hand Gesture Recognition")
st.write("Upload an image to classify the hand gesture and perform the corresponding operation")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Predict the gesture
    prediction = model.predict(preprocessed_image)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    
    gesture = class_labels[class_index]
    operation = gesture_operations[gesture]
    
    st.write(f'Predicted Class: {gesture}, Confidence: {confidence:.2f}')
    st.write(f'Corresponding Operation: {operation}')
