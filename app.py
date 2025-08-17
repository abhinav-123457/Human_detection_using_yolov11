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
def load_yolo_model(model_path, model_type="human"):
    try:
        model = YOLO(model_path)
        logger.info(f"{model_type} YOLO model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load {model_type} YOLO model: {e}")
        raise

# Video processor for YOLO detection (human or face)
class YoloVideoProcessor(VideoProcessorBase):
    def __init__(self, model, conf_threshold, iou_threshold, model_type):
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_type = model_type
        logger.info(f"YoloVideoProcessor initialized for {model_type} with conf_threshold={conf_threshold}, iou_threshold={iou_threshold}")

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
            logger.error(f"Error processing {self.model_type} YOLO frame: {e}")
            raise

# Function to process uploaded images for YOLO
def process_yolo_image(image, model, conf_threshold, iou_threshold, model_type):
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
        logger.error(f"Error processing {model_type} YOLO image: {e}")
        raise

# Auto-adjust thresholds for YOLO
def auto_adjust_thresholds(detection_count, current_conf, current_iou, model_type):
    min_detections = 1
    max_detections = 10
    conf_step = 0.05
    iou_step = 0.05

    if detection_count > max_detections:
        new_conf = min(current_conf + conf_step, 1.0)
        new_iou = max(current_iou - iou_step, 0.1)
        return new_conf, new_iou, f"Increased confidence and decreased IoU for {model_type} to reduce overlapping detections."
    elif detection_count < min_detections:
        new_conf = max(current_conf - conf_step, 0.1)
        new_iou = min(current_iou + iou_step, 1.0)
        return new_conf, new_iou, f"Decreased confidence and increased IoU for {model_type} to increase detections."
    else:
        return current_conf, current_iou, f"Thresholds unchanged for {model_type}: detection count within range."

def main():
    st.set_page_config(page_title="YOLOv11 Human & Face Detection", layout="wide")
    
    st.title("Real-Time Human and Face Detection with YOLOv11")
    st.markdown("""
        This app performs real-time human and face detection using trained YOLOv11 models.
        Use your webcam for live detection or upload an image for static analysis.
        Models are cached for faster loading. Auto-adjustment of thresholds is available.
    """)
    
    # Initialize session state with defaults
    st.session_state.setdefault("human_conf_threshold", 0.10)
    st.session_state.setdefault("human_iou_threshold", 0.45)
    st.session_state.setdefault("face_conf_threshold", 0.10)
    st.session_state.setdefault("face_iou_threshold", 0.45)
    st.session_state.setdefault("human_auto_adjust", False)
    st.session_state.setdefault("face_auto_adjust", False)
    st.session_state.setdefault("adjustment_message", "")
    st.session_state.setdefault("webcam_error", "")

    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    # Human YOLO model path input
    human_model_path = st.sidebar.text_input(
        "Human YOLO Model Path",
        value="yolo11n_human_detection_final.pt",
        help="Path to your trained YOLOv11 human detection model (.pt file)"
    )
    
    # Face YOLO model path input
    face_model_path = st.sidebar.text_input(
        "Face YOLO Model Path",
        value="yolo11n_face_detection.pt",
        help="Path to your trained YOLOv11 face detection model (.pt file)"
    )
    
    # Auto-adjustment toggles
    st.session_state.human_auto_adjust = st.sidebar.checkbox(
        "Enable Auto-Adjustment for Human Detection Thresholds",
        value=st.session_state.human_auto_adjust,
        help="Automatically adjust human detection Confidence and IoU thresholds"
    )
    st.session_state.face_auto_adjust = st.sidebar.checkbox(
        "Enable Auto-Adjustment for Face Detection Thresholds",
        value=st.session_state.face_auto_adjust,
        help="Automatically adjust face detection Confidence and IoU thresholds"
    )
    
    # Human detection thresholds
    human_conf_threshold = st.sidebar.slider(
        "Human Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.human_conf_threshold,
        step=0.05,
        help="Filter human detections below this confidence score",
        disabled=st.session_state.human_auto_adjust
    )
    human_iou_threshold = st.sidebar.slider(
        "Human IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.human_iou_threshold,
        step=0.05,
        help="IoU threshold for human Non-Max Suppression",
        disabled=st.session_state.human_auto_adjust
    )
    
    # Face detection thresholds
    face_conf_threshold = st.sidebar.slider(
        "Face Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.face_conf_threshold,
        step=0.05,
        help="Filter face detections below this confidence score",
        disabled=st.session_state.face_auto_adjust
    )
    face_iou_threshold = st.sidebar.slider(
        "Face IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.face_iou_threshold,
        step=0.05,
        help="IoU threshold for face Non-Max Suppression",
        disabled=st.session_state.face_auto_adjust
    )
    
    # Update session state if manual sliders are used
    if not st.session_state.human_auto_adjust:
        st.session_state.human_conf_threshold = human_conf_threshold
        st.session_state.human_iou_threshold = human_iou_threshold
    if not st.session_state.face_auto_adjust:
        st.session_state.face_conf_threshold = face_conf_threshold
        st.session_state.face_iou_threshold = face_iou_threshold
    
    # Load models
    human_model = None
    face_model = None
    if human_model_path and os.path.exists(human_model_path):
        try:
            human_model = load_yolo_model(human_model_path, model_type="human")
            st.sidebar.success("Human YOLO model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading human YOLO model: {e}")
            logger.error(f"Human YOLO model loading failed: {e}")
    else:
        st.sidebar.warning("Please provide a valid human YOLO model path.")
    
    if face_model_path and os.path.exists(face_model_path):
        try:
            face_model = load_yolo_model(face_model_path, model_type="face")
            st.sidebar.success("Face YOLO model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading face YOLO model: {e}")
            logger.error(f"Face YOLO model loading failed: {e}")
    else:
        st.sidebar.warning("Please provide a valid face YOLO model path.")
    
    # Display adjustment message
    if st.session_state.adjustment_message:
        st.sidebar.info(st.session_state.adjustment_message)
    
    # Tabs for detection modes
    tab1, tab2, tab3 = st.tabs(["Human Webcam Detection", "Face Webcam Detection", "Image Upload"])
    
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
        
        if human_model:
            try:
                webrtc_ctx = webrtc_streamer(
                    key="yolo-human-detection",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=lambda: YoloVideoProcessor(
                        human_model,
                        st.session_state.human_conf_threshold,
                        st.session_state.human_iou_threshold,
                        model_type="human"
                    ),
                    media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}}, "audio": False},
                    async_processing=True,
                )
                
                if webrtc_ctx.state.playing:
                    st.info("Human webcam detection is active. Adjust settings in the sidebar.")
                    st.session_state.webcam_error = ""
                else:
                    st.session_state.webcam_error = "Human webcam stream not active. Click 'Start' or check your webcam."
            except Exception as e:
                st.session_state.webcam_error = f"Failed to initialize human webcam: {str(e)}"
                logger.error(f"Human YOLO WebRTC initialization failed: {e}")
                st.error(st.session_state.webcam_error)
        else:
            st.warning("Human detection model not loaded. Please check the model path.")
    
    with tab2:
        st.header("YOLO Webcam Face Detection")
        st.write("Click 'Start' to begin real-time face detection using your webcam.")
        
        if st.session_state.webcam_error:
            st.error(f"Webcam Error: {st.session_state.webcam_error}")
        
        if face_model:
            try:
                webrtc_ctx = webrtc_streamer(
                    key="yolo-face-detection",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=lambda: YoloVideoProcessor(
                        face_model,
                        st.session_state.face_conf_threshold,
                        st.session_state.face_iou_threshold,
                        model_type="face"
                    ),
                    media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 15}}, "audio": False},
                    async_processing=True,
                )
                
                if webrtc_ctx.state.playing:
                    st.info("Face webcam detection is active. Adjust settings in the sidebar.")
                    st.session_state.webcam_error = ""
                else:
                    st.session_state.webcam_error = "Face webcam stream not active. Click 'Start' or check your webcam."
            except Exception as e:
                st.session_state.webcam_error = f"Failed to initialize face webcam: {str(e)}"
                logger.error(f"Face YOLO WebRTC initialization failed: {e}")
                st.error(st.session_state.webcam_error)
        else:
            st.warning("Face detection model not loaded. Please check the model path.")
    
    with tab3:
        st.header("Image Upload")
        detection_mode = st.radio("Select Detection Mode", ["Human Detection", "Face Detection"])
        uploaded_file = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                st.write("Processing image...")
                if detection_mode == "Human Detection":
                    if human_model:
                        detected_image, detection_count = process_yolo_image(
                            image,
                            human_model,
                            st.session_state.human_conf_threshold,
                            st.session_state.human_iou_threshold,
                            model_type="human"
                        )
                        caption = f"Detected Humans ({detection_count} detections)"
                        
                        if st.session_state.human_auto_adjust:
                            new_conf, new_iou, message = auto_adjust_thresholds(
                                detection_count,
                                st.session_state.human_conf_threshold,
                                st.session_state.human_iou_threshold,
                                model_type="human"
                            )
                            st.session_state.human_conf_threshold = new_conf
                            st.session_state.human_iou_threshold = new_iou
                            st.session_state.adjustment_message = message
                    else:
                        st.error("Human detection model not loaded. Please check the model path.")
                        return
                else:
                    if face_model:
                        detected_image, detection_count = process_yolo_image(
                            image,
                            face_model,
                            st.session_state.face_conf_threshold,
                            st.session_state.face_iou_threshold,
                            model_type="face"
                        )
                        caption = f"Detected Faces ({detection_count} faces)"
                        
                        if st.session_state.face_auto_adjust:
                            new_conf, new_iou, message = auto_adjust_thresholds(
                                detection_count,
                                st.session_state.face_conf_threshold,
                                st.session_state.face_iou_threshold,
                                model_type="face"
                            )
                            st.session_state.face_conf_threshold = new_conf
                            st.session_state.face_iou_threshold = new_iou
                            st.session_state.adjustment_message = message
                    else:
                        st.error("Face detection model not loaded. Please check the model path.")
                        return
                
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
