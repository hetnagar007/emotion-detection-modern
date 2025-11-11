
Emotion Detection — Modern Demo (Lightweight)

What:
- Demo app that detects faces and predicts an emotion label using a simple heuristic.
- Works out-of-the-box without heavy ML model downloads.

Run:
1. pip install -r requirements.txt
2. streamlit run app.py

Notes:
- Webcam demo requires camera permission.
- For production, replace src/detector.predict_emotion_from_face with a trained model.
