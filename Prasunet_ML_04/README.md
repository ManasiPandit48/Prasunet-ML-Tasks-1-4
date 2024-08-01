# Hand Gesture Recognition System

This repository contains a hand gesture recognition system that accurately identifies and classifies different hand gestures from image data, enabling intuitive human-computer interaction and gesture-based control systems.

## Features

- **Hand Gesture Recognition**: Utilizes a Convolutional Neural Network (CNN) to classify hand gestures.
- **Live Webcam Integration**: Real-time gesture recognition using a webcam.
- **System Operations**: Executes system tasks such as volume control and window management based on recognized gestures.

## Requirements

- Python 3.x
- TensorFlow
- OpenCV
- PyAutoGUI
- Streamlit (for GUI-based testing)

Install the required packages:
```sh
pip install -r requirements.txt
```

## Dataset

We have used https://www.kaggle.com/datasets/gti-upm/leapgestrecog dataset for training this model.

## Usage

1. **Train the Model**:
    - Open the Jupyter Notebook `Hand_Gesture_Recognition_Training.ipynb`.
    - Follow the steps to preprocess the data, perform EDA, train the model, and save the trained model.

2. **Test the Model**:
    - **GUI-Based Testing**:
        ```sh
        streamlit run image_test.py
        ```
    - **Live Webcam Testing**:
        ```sh
        python webcam_test.py
