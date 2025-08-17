import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RTC configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Cache the model loading
@st.cache_resource
def load_model(model_path="yolo11n_human_detection_final.pt"):
    try:
        return YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Define a video frame processor class for WebRTC
class VideoProcessor:
    def __init__(self, model, conf_threshold, iou_threshold):
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        logger.info(f"VideoProcessor initialized with conf_threshold={conf_threshold}, iou_threshold={iou_threshold}")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
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
        except Exception as e:
            logger.error(f"Error processing webcam frame: {e}")
            raise

def main():
    st.set_page_config(page_title="YOLOv11 Human Detection", layout="wide")
    
    st.title("🎥 Real-Time Human Detection in Thermal Feed")
    st.markdown("""
        This app uses your webcam to detect humans in real-time using a trained YOLOv11 model.
        The model is cached for faster loading. Adjust settings in the sidebar.
    """)
    
    # Initialize session state for thresholds and error handling
    if 'conf_threshold' not in st.session_state:
        st.session_state.conf_threshold = 0.10  # Fixed default Confidence Threshold
    if 'iou_threshold' not in st.session_state:
        st.session_state.iou_threshold = 0.45   # Default IoU Threshold for NMS
    if 'webcam_error' not in st.session_state:
        st.session_state.webcam_error = ""

    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path",
        value="best.pt",
        help="Path to your trained YOLOv11 model (.pt file)"
    )
    
    # Confidence and IoU thresholds
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.conf_threshold,
        step=0.05,
        help="Filter detections below this confidence score"
    )
    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.iou_threshold,
        step=0.05,
        help="Intersection over Union threshold for Non-Max Suppression"
    )
    
    # Update session state with slider values
    st.session_state.conf_threshold = conf_threshold
    st.session_state.iou_threshold = iou_threshold
    
    # Load model with caching
    model = None
    if model_path:
        try:
            model = load_model(model_path)
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
            logger.error(f"Model loading failed: {e}")
            return
    else:
        st.sidebar.warning("Please provide a valid model path.")
        return
    
    # Display webcam error if any
    if st.session_state.webcam_error:
        st.error(f"Webcam Error: {st.session_state.webcam_error}")
        st.markdown("""
            **Troubleshooting Tips**:
            - Ensure your webcam is connected and accessible.
            - Check browser permissions for camera access.
            - Try a different browser (Chrome or Firefox recommended).
            - Verify that the STUN server is accessible.
        """)
    
    # Checkbox to start webcam
    run = st.checkbox("Start Webcam")
    
    # Placeholder for webcam feed
    FRAME_WINDOW = st.image([])
    
    if run:
        try:
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
                st.session_state.webcam_error = ""  # Clear error if webcam is working
            else:
                st.session_state.webcam_error = "Webcam stream not active. Click 'Start' or check your webcam."
                st.error(st.session_state.webcam_error)
        
        except Exception as e:
            st.session_state.webcam_error = f"Failed to initialize webcam: {str(e)}"
            logger.error(f"WebRTC initialization failed: {e}")
            st.error(st.session_state.webcam_error)

if __name__ == "__main__":
    main()
