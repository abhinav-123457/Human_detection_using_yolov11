import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Real-Time Face Detection")

class FaceDetection(VideoTransformerBase):
    def __init__(self):
        # Load the Haar Cascade once
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return img

webrtc_streamer(key="face-detection", video_transformer_factory=FaceDetection)
