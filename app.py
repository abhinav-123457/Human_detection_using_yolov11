import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os

# RTC configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Cache the model loading
@st.cache_resource
def load_model(model_path="yolo11n_human_detection_final.pt"):
    return YOLO(model_path)

# Define a video frame processor class
class VideoProcessor:
    def __init__(self, model, conf_threshold, iou_threshold):
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        # Run YOLO inference
        results = self.model.predict(
            img,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        # Draw bounding boxes with only "human" label
        annotated_img = results[0].plot(labels=True, conf=False)
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

# Function to process uploaded images
def process_image(image, model, conf_threshold, iou_threshold):
    img_array = np.array(image)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    results = model.predict(
        img_array,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False
    )
    # Draw bounding boxes with only "human" label
    annotated_img = results[0].plot(labels=True, conf=False)
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_img)

def main():
    st.set_page_config(page_title="YOLOv11 Human Detection", layout="wide")
    
    st.title("Real-Time Human Detection with YOLOv11")
    st.markdown("""
        This app performs real-time human detection using a trained YOLOv11 model.
        Use your webcam for live detection or upload an image for static analysis.
        The model is cached for faster loading.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path",
        value="yolo11n_human_detection_final.pt",
        help="Path to your trained YOLOv11 model (.pt file)"
    )
    
    # Confidence and IoU thresholds
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Filter detections below this confidence score"
    )
    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="Intersection over Union threshold for Non-Max Suppression"
    )
    
    # Load model with caching
    model = None
    if model_path and os.path.exists(model_path):
        try:
            model = load_model(model_path)
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
            return
    else:
        st.sidebar.warning("Please provide a valid model path.")
        return
    
    # Tabs for webcam and image upload
    tab1, tab2 = st.tabs(["Webcam Detection", "Image Upload"])
    
    with tab1:
        st.header("Webcam Detection")
        st.write("Click 'Start' to begin real-time human detection using your webcam.")
        
        webrtc_ctx = webrtc_streamer(
            key="human-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: VideoProcessor(model, conf_threshold, iou_threshold),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if webrtc_ctx.state.playing:
            st.info("Webcam detection is active. Adjust settings in the sidebar.")
    
    with tab2:
        st.header("Image Upload")
        uploaded_file = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process and display detected image
            st.write("Processing image...")
            detected_image = process_image(image, model, conf_threshold, iou_threshold)
            st.image(detected_image, caption="Detected Humans", use_column_width=True)
            
            # Convert detected image to bytes for download
            img_buffer = BytesIO()
            detected_image.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            
            # Download button for processed image
            st.download_button(
                label="Download Detected Image",
                data=img_buffer,
                file_name="detected_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
