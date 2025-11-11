
import cv2
import numpy as np
from pathlib import Path

# Use OpenCV's Haar cascade for face detection (bundled with OpenCV)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

EMOTIONS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def detect_faces(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    # returns list of (x,y,w,h) and grayscale image
    return faces, gray

def predict_emotion_from_face(face_gray):
    """
    Simple heuristic predictor for demo:
    - mean brightness high -> Happy
    - mean brightness low -> Sad
    - high standard deviation (contrast) -> Surprise
    - else Neutral
    Returns (label, confidence)
    """
    if face_gray is None or face_gray.size == 0:
        return "Neutral", 0.0
    mean = float(np.mean(face_gray))
    std = float(np.std(face_gray))
    # normalize values roughly into confidence-like score
    if std > 60:
        return "Surprise", min(0.99, (std/150.0))
    if mean > 150:
        return "Happy", min(0.95, (mean-100)/155.0)
    if mean < 80:
        return "Sad", min(0.95, (80-mean)/80.0)
    # fallback
    return "Neutral", 0.6
