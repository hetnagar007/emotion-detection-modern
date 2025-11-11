
import cv2
import numpy as np

def read_image_from_uploaded(uploaded_file):
    # Streamlit uploads provide a file-like object; OpenCV needs numpy array
    file_bytes = uploaded_file.read()
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img
