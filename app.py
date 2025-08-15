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
    return Image.fromarray(annotated_img), len(results[0].boxes)  # Return image and detection count

# Auto-adjust thresholds based on detection count
def auto_adjust_thresholds(detection_count, current_conf, current_iou):
    # Define thresholds for adjustment
    min_detections = 1  # Minimum desired detections
    max_detections = 10  # Maximum desired detections
    conf_step = 0.05
    iou_step = 0.05

    # Adjust confidence threshold
    if detection_count > max_detections:
        # Too many detections, increase confidence threshold to be stricter
        new_conf = min(current_conf + conf_step, 1.0)
        new_iou = current_iou
        return new_conf, new_iou, "Increased confidence threshold to reduce detections."
    elif detection_count < min_detections:
        # Too few detections, decrease confidence threshold to be more lenient
        new_conf = max(current_conf - conf_step, 0.1)
        new_iou = current_iou
        return new_conf, new_iou, "Decreased confidence threshold to increase detections."
    else:
        # Detection count is within acceptable range, no adjustment needed
        return current_conf, current_iou, "Thresholds unchanged: detection count within range."

def main():
    st.set_page_config(page_title="YOLOv11 Human Detection", layout="wide")
    
    st.title("Real-Time Human Detection with YOLOv11")
    st.markdown("""
        This app performs real-time human detection using a trained YOLOv11 model.
        Use your webcam for live detection or upload an image for static analysis.
        The model is cached for faster loading. Auto-adjustment of thresholds is available.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    # Model path input
    model_path = st.sidebar.text_input(
        "Model Path",
        value="yolo11n_human_detection_final.pt",
        help="Path to your trained YOLOv11 model (.pt file)"
    )
    
    # Initialize session state for thresholds
    if 'conf_threshold' not in st.session_state:
        st.session_state.conf_threshold = 0.5
    if 'iou_threshold' not in st.session_state:
        st.session_state.iou_threshold = 0.45
    if 'auto_adjust' not in st.session_state:
        st.session_state.auto_adjust = False
    if 'adjustment_message' not in st.session_state:
        st.session_state.adjustment_message = ""

    # Auto-adjustment toggle
    st.session_state.auto_adjust = st.sidebar.checkbox(
        "Enable Auto-Adjustment of Thresholds",
        value=st.session_state.auto_adjust,
        help="Automatically adjust Confidence and IoU thresholds based on detection count"
    )
    
    # Confidence and IoU thresholds (manual sliders, disabled if auto-adjust is on)
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.conf_threshold,
        step=0.05,
        help="Filter detections below this confidence score",
        disabled=st.session_state.auto_adjust
    )
    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.iou_threshold,
        step=0.05,
        help="Intersection over Union threshold for Non-Max Suppression",
        disabled=st.session_state.auto_adjust
    )
    
    # Update session state if manual sliders are used
    if not st.session_state.auto_adjust:
        st.session_state.conf_threshold = conf_threshold
        st.session_state.iou_threshold = iou_threshold
    
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
    
    # Display adjustment message
    if st.session_state.adjustment_message:
        st.sidebar.info(st.session_state.adjustment_message)
    
    # Tabs for webcam and image upload
    tab1, tab2 = st.tabs(["Webcam Detection", "Image Upload"])
    
    with tab1:
        st.header("Webcam Detection")
        st.write("Click 'Start' to begin real-time human detection using your webcam.")
        
        webrtc_ctx = webrtc_streamer(
            key="human-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: VideoProcessor(model, st.session_state.conf_threshold, st.session_state.iou_threshold),
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
            detected_image, detection_count = process_image(image, model, st.session_state.conf_threshold, st.session_state.iou_threshold)
            st.image(detected_image, caption=f"Detected Humans ({detection_count} detections)", use_column_width=True)
            
            # Auto-adjust thresholds if enabled
            if st.session_state.auto_adjust:
                new_conf, new_iou, message = auto_adjust_thresholds(
                    detection_count,
                    st.session_state.conf_threshold,
                    st.session_state.iou_threshold
                )
                st.session_state.conf_threshold = new_conf
                st.session_state.iou_threshold = new_iou
                st.session_state.adjustment_message = message
            
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
