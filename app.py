import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
import av
import os
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RTC configuration with multiple ICE servers for reliability
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]
})

# Cache the YOLO model loading
@st.cache_resource
def load_yolo_model(model_path="yolo11n_human_detection_final.pt"):
    try:
        return YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        raise

# Cache the Haar Cascade classifier loading
@st.cache_resource
def load_haar_cascade():
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            raise FileNotFoundError("Haar Cascade file not found")
        return cv2.CascadeClassifier(cascade_path)
    except Exception as e:
        logger.error(f"Failed to load Haar Cascade: {e}")
        raise

# Video processor for YOLO human detection
class YoloVideoProcessor(VideoProcessorBase):
    def __init__(self, model, conf_threshold, iou_threshold):
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        logger.info(f"YoloVideoProcessor initialized with conf_threshold={conf_threshold}, iou_threshold={iou_threshold}")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            results = self.model.predict(
                img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device="cpu"  # Use CPU to avoid GPU memory issues; switch to "cuda" if GPU is available
            )
            annotated_img = results[0].plot(labels=True, conf=False)
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        except Exception as e:
            logger.error(f"Error processing YOLO frame: {e}")
            raise

# Video processor for Haar Cascade face detection
class FaceVideoProcessor(VideoProcessorBase):
    def __init__(self, face_cascade, scale_factor=1.1, min_neighbors=5):
        self.face_cascade = face_cascade
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        logger.info(f"FaceVideoProcessor initialized with scale_factor={scale_factor}, min_neighbors={min_neighbors}")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(30, 30)
            )
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            logger.error(f"Error processing face detection frame: {e}")
            raise

# Function to process uploaded images for YOLO
def process_yolo_image(image, model, conf_threshold, iou_threshold):
    try:
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        results = model.predict(
            img_array,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        annotated_img = results[0].plot(labels=True, conf=False)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_img), len(results[0].boxes)
    except Exception as e:
        logger.error(f"Error processing YOLO image: {e}")
        raise

# Function to process uploaded images for face detection
def process_face_image(image, face_cascade, scale_factor, min_neighbors):
    try:
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
        annotated_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_img), len(faces)
    except Exception as e:
        logger.error(f"Error processing face image: {e}")
        raise

# Auto-adjust thresholds for YOLO
def auto_adjust_thresholds(detection_count, current_conf, current_iou):
    min_detections = 1
    max_detections = 10
    conf_step = 0.05
    iou_step = 0.05

    if detection_count > max_detections:
        new_conf = min(current_conf + conf_step, 1.0)
        new_iou = max(current_iou - iou_step, 0.1)
        return new_conf, new_iou, "Increased confidence and decreased IoU to reduce overlapping detections."
    elif detection_count < min_detections:
        new_conf = max(current_conf - conf_step, 0.1)
        new_iou = min(current_iou + iou_step, 1.0)
        return new_conf, new_iou, "Decreased confidence and increased IoU to increase detections."
    else:
        return current_conf, current_iou, "Thresholds unchanged: detection count within range."

def main():
    st.set_page_config(page_title="YOLOv11 & Face Detection", layout="wide")
    
    st.title("Real-Time Human and Face Detection")
    st.markdown("""
        This app performs real-time human detection using YOLOv11 and face detection using Haar Cascade.
        Use your webcam for live detection or upload an image for static analysis.
        The models are cached for faster loading. Auto-adjustment of thresholds is available for YOLO.
    """)
    
    # Initialize session state
    if 'conf_threshold' not in st.session_state:
        st.session_state.conf_threshold = 0.10
    if 'iou_threshold' not in st.session_state:
        st.session_state.iou_threshold = 0.45
    if 'auto_adjust' not in st.session_state:
        st.session_state.auto_adjust = False
    if 'adjustment_message' not in st.session_state:
        st.session_state.adjustment_message = ""
    if 'webcam_error' not in st.session_state:
        st.session_state.webcam_error = ""
    if 'face_scale_factor' not in st.session_state:
        st.session_state.face_scale_factor = 1.1
    if 'face_min_neighbors' not in st.session_state:
        st.session_state.face_min_neighbors = 5

    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    # YOLO model path input
    model_path = st.sidebar.text_input(
        "YOLO Model Path",
        value="yolo11n_human_detection_final.pt",
        help="Path to your trained YOLOv11 model (.pt file)"
    )
    
    # Auto-adjustment toggle for YOLO
    st.session_state.auto_adjust = st.sidebar.checkbox(
        "Enable Auto-Adjustment for YOLO Thresholds",
        value=st.session_state.auto_adjust,
        help="Automatically adjust YOLO Confidence and IoU thresholds"
    )
    
    # YOLO thresholds
    conf_threshold = st.sidebar.slider(
        "YOLO Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.conf_threshold,
        step=0.05,
        help="Filter YOLO detections below this confidence score",
        disabled=st.session_state.auto_adjust
    )
    iou_threshold = st.sidebar.slider(
        "YOLO IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.iou_threshold,
        step=0.05,
        help="IoU threshold for YOLO Non-Max Suppression",
        disabled=st.session_state.auto_adjust
    )
    
    # Face detection parameters
    face_scale_factor = st.sidebar.slider(
        "Face Detection Scale Factor",
        min_value=1.05,
        max_value=1.5,
        value=st.session_state.face_scale_factor,
        step=0.05,
        help="Scale factor for Haar Cascade face detection"
    )
    face_min_neighbors = st.sidebar.slider(
        "Face Detection Min Neighbors",
        min_value=3,
        max_value=10,
        value=st.session_state.face_min_neighbors,
        step=1,
        help="Minimum neighbors for Haar Cascade face detection"
    )
    
    # Update session state
    if not st.session_state.auto_adjust:
        st.session_state.conf_threshold = conf_threshold
        st.session_state.iou_threshold = iou_threshold
    st.session_state.face_scale_factor = face_scale_factor
    st.session_state.face_min_neighbors = face_min_neighbors
    
    # Load models
    yolo_model = None
    face_cascade = None
    if model_path and os.path.exists(model_path):
        try:
            yolo_model = load_yolo_model(model_path)
            st.sidebar.success("YOLO model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading YOLO model: {e}")
            logger.error(f"YOLO model loading failed: {e}")
            return
    else:
        st.sidebar.warning("Please provide a valid YOLO model path.")
    
    try:
        face_cascade = load_haar_cascade()
        st.sidebar.success("Haar Cascade loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading Haar Cascade: {e}")
        logger.error(f"Haar Cascade loading failed: {e}")
        return
    
    # Display adjustment message
    if st.session_state.adjustment_message:
        st.sidebar.info(st.session_state.adjustment_message)
    
    # Tabs for detection modes
    tab1, tab2, tab3 = st.tabs(["YOLO Webcam Detection", "Face Webcam Detection", "Image Upload"])
    
    with tab1:
        st.header("YOLO Webcam Human Detection")
        st.write("Click 'Start' to begin real-time human detection using your webcam.")
        
        if st.session_state.webcam_error:
            st.error(f"Webcam Error: {st.session_state.webcam_error}")
            st.markdown("""
                **Troubleshooting Tips**:
                - Ensure your webcam is connected and accessible.
                - Check browser permissions for camera access.
                - Try a different browser (Chrome or Firefox recommended).
                - Verify that the STUN server is accessible.
            """)
        
        try:
            webrtc_ctx = webrtc_streamer(
                key="yolo-human-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=lambda: YoloVideoProcessor(
                    yolo_model,
                    st.session_state.conf_threshold,
                    st.session_state.iou_threshold
                ),
                media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}}, "audio": False},
                async_processing=True,
            )
            
            if webrtc_ctx.state.playing:
                st.info("YOLO webcam detection is active. Adjust settings in the sidebar.")
                st.session_state.webcam_error = ""
            else:
                st.session_state.webcam_error = "YOLO webcam stream not active. Click 'Start' or check your webcam."
        
        except Exception as e:
            st.session_state.webcam_error = f"Failed to initialize YOLO webcam: {str(e)}"
            logger.error(f"YOLO WebRTC initialization failed: {e}")
            st.error(st.session_state.webcam_error)
    
    with tab2:
        st.header("Face Webcam Detection")
        st.write("Click 'Start' to begin real-time face detection using your webcam.")
        
        if st.session_state.webcam_error:
            st.error(f"Webcam Error: {st.session_state.webcam_error}")
        
        try:
            webrtc_ctx = webrtc_streamer(
                key="face-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=lambda: FaceVideoProcessor(
                    face_cascade,
                    st.session_state.face_scale_factor,
                    st.session_state.face_min_neighbors
                ),
                media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}}, "audio": False},
                async_processing=True,
            )
            
            if webrtc_ctx.state.playing:
                st.info("Face detection webcam is active. Adjust settings in the sidebar.")
                st.session_state.webcam_error = ""
            else:
                st.session_state.webcam_error = "Face detection webcam stream not active. Click 'Start' or check your webcam."
        
        except Exception as e:
            st.session_state.webcam_error = f"Failed to initialize face detection webcam: {str(e)}"
            logger.error(f"Face WebRTC initialization failed: {e}")
            st.error(st.session_state.webcam_error)
    
    with tab3:
        st.header("Image Upload")
        detection_mode = st.radio("Select Detection Mode", ["YOLO Human Detection", "Face Detection"])
        uploaded_file = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                st.write("Processing image...")
                if detection_mode == "YOLO Human Detection":
                    detected_image, detection_count = process_yolo_image(
                        image,
                        yolo_model,
                        st.session_state.conf_threshold,
                        st.session_state.iou_threshold
                    )
                    caption = f"Detected Humans ({detection_count} detections)"
                    
                    if st.session_state.auto_adjust:
                        new_conf, new_iou, message = auto_adjust_thresholds(
                            detection_count,
                            st.session_state.conf_threshold,
                            st.session_state.iou_threshold
                        )
                        st.session_state.conf_threshold = new_conf
                        st.session_state.iou_threshold = new_iou
                        st.session_state.adjustment_message = message
                else:
                    detected_image, detection_count = process_face_image(
                        image,
                        face_cascade,
                        st.session_state.face_scale_factor,
                        st.session_state.face_min_neighbors
                    )
                    caption = f"Detected Faces ({detection_count} faces)"
                
                st.image(detected_image, caption=caption, use_column_width=True)
                
                img_buffer = BytesIO()
                detected_image.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                
                st.download_button(
                    label="Download Detected Image",
                    data=img_buffer,
                    file_name="detected_image.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Error processing image: {e}")
                logger.error(f"Image processing failed: {e}")

if __name__ == "__main__":
    main()
