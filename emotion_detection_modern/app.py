
import streamlit as st
import cv2, time
from pathlib import Path
from src.detector import detect_faces, predict_emotion_from_face
from src.utils import read_image_from_uploaded
import numpy as np

st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="centered")

st.markdown("<h1 style='text-align:center'>🧠 Emotion Detection — Demo</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Lightweight demo using OpenCV face detection and a heuristic emotion predictor. Ideal for quick demos and portfolio.</p>", unsafe_allow_html=True)
st.write("---")

col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Upload an image")
    uploaded = st.file_uploader("Choose a photo with a clear face", type=["jpg","jpeg","png"])
    if uploaded is not None:
        img = read_image_from_uploaded(uploaded)
        if img is None:
            st.error("Couldn't read image. Try another file.")
        else:
            faces, gray = detect_faces(img)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_column_width=True)
            if len(faces) == 0:
                st.warning("No faces detected. Try a different image or angle.")
            else:
                st.success(f"Detected {len(faces)} face(s).")
                for (x,y,w,h) in faces:
                    face_gray = gray[y:y+h, x:x+w]
                    label, conf = predict_emotion_from_face(face_gray)
                    st.write(f"Emotion: **{label}**  — Confidence: {conf:.2f}")
                    # show face crop
                    face_bgr = img[y:y+h, x:x+w]
                    st.image(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB), width=150)
with col2:
    st.subheader("Live webcam demo")
    st.write("Start the webcam to try real-time detection. (Requires OpenCV access to your camera)")
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to open webcam. Check permissions or camera index.")
        else:
            t0 = time.time()
            # capture for a few seconds
            for i in range(150):  # ~ 5 seconds at 30fps max loop, but we process fewer frames
                ret, frame = cap.read()
                if not ret:
                    break
                faces, gray = detect_faces(frame)
                for (x,y,w,h) in faces:
                    face_gray = gray[y:y+h, x:x+w]
                    label, conf = predict_emotion_from_face(face_gray)
                    # draw rectangle and label
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                # convert to RGB for st.image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, channels="RGB")
            cap.release()
            st.success("Webcam demo finished.")

st.write("---")
st.markdown("**Notes:** This is a demo — for production use, replace heuristic predictor with a trained CNN (FER2013).")
