import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import os
import logging
import time
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RTC configuration with multiple STUN servers
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]}
    ]
})

# Cache the model loading
@st.cache_resource
def load_model(model_path="yolo11n_human_detection_final.pt"):
    try:
        model = YOLO(model_path)
        if model.device.type == 'cuda':
            model = model.half()
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# WebRTC video processor
class VideoProcessor:
    def __init__(self, model, conf_threshold, iou_threshold, frame_skip=2):
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.last_fps = 0
        self.last_time = time.time()
        logger.info(f"VideoProcessor initialized with conf_threshold={conf_threshold}, iou_threshold={iou_threshold}, frame_skip={frame_skip}")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.last_fps = self.frame_count / (current_time - self.last_time)
                self.frame_count = 0
                self.last_time = current_time

            if self.frame_count % self.frame_skip != 0:
                return frame

            img = frame.to_ndarray(format="bgr24")
            results = self.model.predict(
                img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                half=True if self.model.device.type == 'cuda' else False
            )
            annotated_img = results[0].plot(labels=True, conf=False)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            annotated_img = cv2.filter2D(annotated_img, -1, kernel)
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        except Exception as e:
            logger.error(f"Error processing webcam frame: {e}")
            raise

# Process a single frame (OpenCV)
def process_frame(img, model, conf_threshold, iou_threshold):
    try:
        results = model.predict(
            img,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            half=True if model.device.type == 'cuda' else False
        )
        annotated_img = results[0].plot(labels=True, conf=False)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        annotated_img = cv2.filter2D(annotated_img, -1, kernel)
        detection_count = len(results[0].boxes)
        return annotated_img, detection_count
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        raise

# Process uploaded images
def process_image(image, model, conf_threshold, iou_threshold):
    try:
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        results = model.predict(
            img_array,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
            half=True if model.device.type == 'cuda' else False
        )
        annotated_img = results[0].plot(labels=True, conf=False)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_img), len(results[0].boxes)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

# Auto-adjust thresholds
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

# Function to get available WebRTC devices
async def get_webrtc_devices():
    try:
        import webrtcvad
        devices = await webrtcvad.get_device_list()
        video_devices = [d['label'] or f"Camera {i}" for i, d in enumerate(devices) if d['kind'] == 'videoinput']
        return video_devices
    except Exception as e:
        logger.error(f"Error fetching WebRTC devices: {e}")
        return ["Default Camera"]

def main():
    st.set_page_config(page_title="YOLOv11 Human Detection", layout="wide")
    
    st.title("Real-Time Human Detection with YOLOv11")
    st.markdown("""
        This app performs real-time human detection using a trained YOLOv11 model.
        Use your webcam (via WebRTC or OpenCV) or upload an image.
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
    if 'frame_skip' not in st.session_state:
        st.session_state.frame_skip = 2
    if 'use_webrtc' not in st.session_state:
        st.session_state.use_webrtc = True
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'webcam_index' not in st.session_state:
        st.session_state.webcam_index = 0
    if 'webrtc_device' not in st.session_state:
        st.session_state.webrtc_device = "Default Camera"

    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value="yolo11n_human_detection_final.pt",
        help="Path to your trained YOLOv11 model (.pt file)"
    )
    
    st.session_state.use_webrtc = st.sidebar.checkbox(
        "Use WebRTC (uncheck for OpenCV)",
        value=st.session_state.use_webrtc,
        help="WebRTC for browser-based access, OpenCV for lower latency (local only)"
    )
    
    st.session_state.auto_adjust = st.sidebar.checkbox(
        "Enable Auto-Adjustment of Thresholds",
        value=st.session_state.auto_adjust
    )
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.conf_threshold,
        step=0.05,
        disabled=st.session_state.auto_adjust
    )
    iou_threshold = st.sidebar.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.iou_threshold,
        step=0.05,
        disabled=st.session_state.auto_adjust
    )
    
    st.session_state.frame_skip = st.sidebar.slider(
        "Frame Skip",
        min_value=1,
        max_value=5,
        value=st.session_state.frame_skip,
        step=1
    )

    if not st.session_state.auto_adjust:
        st.session_state.conf_threshold = conf_threshold
        st.session_state.iou_threshold = iou_threshold
    
    # Load model
    model = None
    if model_path and os.path.exists(model_path):
        try:
            model = load_model(model_path)
            st.sidebar.success(f"Model loaded! Device: {model.device.type}")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
            return
    else:
        st.sidebar.warning("Provide a valid model path.")
        return
    
    if st.session_state.adjustment_message:
        st.sidebar.info(st.session_state.adjustment_message)
    
    tab1, tab2 = st.tabs(["Webcam Detection", "Image Upload"])
    
    with tab1:
        st.header("Webcam Detection")
        st.write("Click 'Start' to begin real-time detection.")
        
        if st.session_state.webcam_error:
            st.error(f"Webcam Error: {st.session_state.webcam_error}")
            st.markdown("""
                **Troubleshooting Tips**:
                - **WebRTC**:
                  - Ensure browser permissions allow camera access (Settings > Privacy > Camera).
                  - Select the Logitech C270 in the device dropdown.
                  - Try Chrome/Firefox and disable antivirus webcam protection (e.g., Avast Webcam Shield).
                - **OpenCV**:
                  - Try different webcam indices (0, 1, 2).
                  - Run `pip install opencv-python opencv-contrib-python` to ensure proper backends.
                  - Close apps like Zoom/Skype that may lock the camera.
                  - Update Logitech drivers from https://www.logitech.com/.
                - Check USB connection and try a different port.
                - Adjust 'Frame Skip' to reduce latency.
            """)
        
        start_button = st.button("Start Webcam")
        stop_button = st.button("Stop Webcam")
        frame_placeholder = st.empty()
        
        if start_button:
            st.session_state.webcam_active = True
            st.session_state.webcam_error = ""
        
        if stop_button:
            st.session_state.webcam_active = False
        
        if st.session_state.webcam_active:
            if st.session_state.use_webrtc:
                # Get available WebRTC devices
                try:
                    video_devices = asyncio.run(get_webrtc_devices())
                    st.session_state.webrtc_device = st.selectbox(
                        "Select Webcam",
                        options=video_devices,
                        index=video_devices.index(st.session_state.webrtc_device) if st.session_state.webrtc_device in video_devices else 0
                    )
                except Exception as e:
                    st.session_state.webcam_error = f"Failed to list WebRTC devices: {e}"
                    video_devices = ["Default Camera"]
                
                try:
                    webrtc_ctx = webrtc_streamer(
                        key="human-detection",
                        mode=WebRtcMode.SENDRECV,
                        rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=lambda: VideoProcessor(model, st.session_state.conf_threshold, st.session_state.iou_threshold, st.session_state.frame_skip),
                        media_stream_constraints={
                            "video": {
                                "width": {"ideal": 640},
                                "height": {"ideal": 480},
                                "frameRate": {"ideal": 15},
                                "deviceId": {"exact": st.session_state.webrtc_device}
                            },
                            "audio": False
                        },
                        async_processing=True
                    )
                    
                    if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                        st.info(f"WebRTC active. Device: {st.session_state.webrtc_device}. FPS: {webrtc_ctx.video_processor.last_fps:.1f}")
                        st.session_state.webcam_error = ""
                    else:
                        st.session_state.webcam_error = "WebRTC stream not active. Select Logitech C270 or check permissions."
                        frame_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="No webcam feed")
                
                except Exception as e:
                    st.session_state.webcam_error = f"Failed to initialize WebRTC: {str(e)}"
                    logger.error(f"WebRTC initialization failed: {e}")
                    frame_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="No webcam feed")
            else:
                # OpenCV mode
                st.session_state.webcam_index = st.number_input(
                    "Webcam Index",
                    min_value=0,
                    max_value=5,
                    value=st.session_state.webcam_index,
                    step=1
                )
                cap = cv2.VideoCapture(st.session_state.webcam_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    st.session_state.webcam_error = f"Failed to access webcam at index {st.session_state.webcam_index}. Try a different index."
                    st.session_state.webcam_active = False
                    frame_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="No webcam feed")
                else:
                    try:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        cap.set(cv2.CAP_PROP_FPS, 15)
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                        
                        frame_count = 0
                        last_time = time.time()
                        while st.session_state.webcam_active:
                            ret, frame = cap.read()
                            if not ret:
                                st.session_state.webcam_error = f"Failed to capture frame at index {st.session_state.webcam_index}."
                                st.session_state.webcam_active = False
                                break
                            
                            frame_count += 1
                            if frame_count % st.session_state.frame_skip == 0:
                                annotated_frame, detection_count = process_frame(
                                    frame, model, st.session_state.conf_threshold, st.session_state.iou_threshold
                                )
                                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(annotated_frame_rgb, caption=f"Webcam Feed ({detection_count} detections)")
                                
                                if st.session_state.auto_adjust:
                                    new_conf, new_iou, message = auto_adjust_thresholds(
                                        detection_count,
                                        st.session_state.conf_threshold,
                                        st.session_state.iou_threshold
                                    )
                                    st.session_state.conf_threshold = new_conf
                                    st.session_state.iou_threshold = new_iou
                                    st.session_state.adjustment_message = message
                            
                            current_time = time.time()
                            if current_time - last_time >= 1.0:
                                fps = frame_count / (current_time - last_time)
                                frame_count = 0
                                last_time = current_time
                                st.metric("Approx. FPS", f"{fps:.1f}")
                            
                            time.sleep(0.05)
                            
                    except Exception as e:
                        st.session_state.webcam_error = f"Error processing webcam feed: {str(e)}"
                        st.session_state.webcam_active = False
                        frame_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8), caption="No webcam feed")
                    finally:
                        cap.release()
    
    with tab2:
        st.header("Image Upload")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image")
                
                st.write("Processing image...")
                detected_image, detection_count = process_image(image, model, st.session_state.conf_threshold, st.session_state.iou_threshold)
                st.image(detected_image, caption=f"Detected Humans ({detection_count} detections)")
                
                if st.session_state.auto_adjust:
                    new_conf, new_iou, message = auto_adjust_thresholds(
                        detection_count,
                        st.session_state.conf_threshold,
                        st.session_state.iou_threshold
                    )
                    st.session_state.conf_threshold = new_conf
                    st.session_state.iou_threshold = new_iou
                    st.session_state.adjustment_message = message
                
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

if __name__ == "__main__":
    main()
