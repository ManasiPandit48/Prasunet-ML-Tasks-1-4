import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pyautogui

# Disable PyAutoGUI fail-safe
pyautogui.FAILSAFE = False

# Load the trained model
model = load_model('handG_Rec.h5')

# Class labels
class_labels = ['palm', 'l', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'c', 'down']

# Gesture to operation mapping
gesture_operations = {
    'palm': 'Volume Up',
    'l': 'Volume Down',
    'fist': 'Minimize Window',
    'fist_moved': 'Maximize Window',
    'thumb': 'Close Window',
    'index': 'Open Notepad',
    'ok': 'Mute/Unmute',
    'palm_moved': 'Pause/Play Media',
    'c': 'Open Calculator',
    'down': 'Lock Screen'
}

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (150, 150))  # Resize to the expected input size of the model
    image = image / 255.0  # Normalize the image
    image = img_to_array(image)
    return np.expand_dims(image, axis=0)

# Function to execute the operation based on the gesture
def execute_operation(gesture):
    if gesture == 'palm':
        pyautogui.press('volumeup')
    elif gesture == 'l':
        pyautogui.press('volumedown')
    elif gesture == 'fist':
        pyautogui.hotkey('win', 'down')
    elif gesture == 'fist_moved':
        pyautogui.hotkey('win', 'up')
    elif gesture == 'thumb':
        pyautogui.hotkey('alt', 'f4')
    elif gesture == 'index':
        pyautogui.hotkey('win', 'r')
        pyautogui.write('notepad')
        pyautogui.press('enter')
    elif gesture == 'ok':
        pyautogui.press('volumemute')
    elif gesture == 'palm_moved':
        pyautogui.press('playpause')
    elif gesture == 'c':
        pyautogui.hotkey('win', 'r')
        pyautogui.write('calc')
        pyautogui.press('enter')
    elif gesture == 'down':
        pyautogui.hotkey('win', 'l')

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    preprocessed_frame = preprocess_image(frame)
    
    # Predict the gesture
    prediction = model.predict(preprocessed_frame)
    class_index = np.argmax(prediction)
    gesture = class_labels[class_index]
    
    # Execute the corresponding operation
    execute_operation(gesture)
    
    # Display the frame with the predicted gesture
    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Gesture Recognition', frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
