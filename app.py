import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os
import logging
import time

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache the model loading
@st.cache_resource
def load_model(model_path="yolo11n_human_detection_final.pt"):
    try:
        return YOLO(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Function to detect available webcam indices
def get_available_cameras(max_index=5):
    available_cameras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
            logger.info(f"Webcam found at index {i}")
        else:
            logger.debug(f"No webcam at index {i}")
    return available_cameras

# Function to process a single frame
def process_frame(img, model, conf_threshold, iou_threshold):
    try:
        results = model.predict(
            img,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        annotated_img = results[0].plot(labels=True, conf=False)
        detection_count = len(results[0].boxes)
        return annotated_img, detection_count
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        raise

# Function to process uploaded images
def process_image(image, model, conf_threshold, iou_threshold):
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
        logger.error(f"Error processing image: {e}")
        raise

# Auto-adjust thresholds based on detection count
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
    st.set_page_config(page_title="YOLOv11 Human Detection", layout="wide")
    
    st.title("Real-Time Human Detection with YOLOv11")
    st.markdown("""
        This app performs real-time human detection using a trained YOLOv11 model.
        Use your webcam for live detection or upload an image for static analysis.
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
    if 'webcam_active' not in st.session_state:
        st.session_state.webcam_active = False
    if 'webcam_index' not in st.session_state:
        st.session_state.webcam_index = 0

    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    model_path = st.sidebar.text_input(
        "Model Path",
        value="yolo11n_human_detection_final.pt",
        help="Path to your trained YOLOv11 model (.pt file)"
    )
    
    st.session_state.auto_adjust = st.sidebar.checkbox(
        "Enable Auto-Adjustment of Thresholds",
        value=st.session_state.auto_adjust,
        help="Automatically adjust Confidence and IoU thresholds based on detection count"
    )
    
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
    
    if not st.session_state.auto_adjust:
        st.session_state.conf_threshold = conf_threshold
        st.session_state.iou_threshold = iou_threshold
    
    # Detect available cameras
    if st.sidebar.button("Refresh Camera List"):
        st.session_state.available_cameras = get_available_cameras()
    
    if 'available_cameras' not in st.session_state:
        st.session_state.available_cameras = get_available_cameras()
    
    if not st.session_state.available_cameras:
        st.session_state.webcam_error = "No webcams detected. Please connect a webcam and try again."
    else:
        st.sidebar.selectbox(
            "Select Webcam",
            options=st.session_state.available_cameras,
            index=st.session_state.available_cameras.index(st.session_state.webcam_index) if st.session_state.webcam_index in st.session_state.available_cameras else 0,
            key="webcam_index_select",
            help="Select the webcam index to use"
        )
        st.session_state.webcam_index = st.session_state.webcam_index_select
    
    # Load model
    model = None
    if model_path and os.path.exists(model_path):
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
    
    if st.session_state.adjustment_message:
        st.sidebar.info(st.session_state.adjustment_message)
    
    tab1, tab2 = st.tabs(["Webcam Detection", "Image Upload"])
    
    with tab1:
        st.header("Webcam Detection")
        st.write("Click 'Start Webcam' to begin real-time human detection using your webcam.")
        
        start_button = st.button("Start Webcam")
        stop_button = st.button("Stop Webcam")
        
        frame_placeholder = st.empty()
        
        # Fallback placeholder image
        placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder_image = cv2.putText(
            placeholder_image,
            "No webcam feed available",
            (50, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        if start_button:
            st.session_state.webcam_active = True
            st.session_state.webcam_error = ""
        
        if stop_button:
            st.session_state.webcam_active = False
        
        if st.session_state.webcam_active:
            cap = cv2.VideoCapture(st.session_state.webcam_index)
            if not cap.isOpened():
                st.session_state.webcam_error = f"Failed to access webcam at index {st.session_state.webcam_index}. Try a different index or check webcam connection."
                st.session_state.webcam_active = False
                logger.error(f"Failed to open webcam at index {st.session_state.webcam_index}")
                frame_placeholder.image(placeholder_image, caption="No webcam feed", use_column_width=True)
            else:
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution for performance
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                    
                    while st.session_state.webcam_active:
                        ret, frame = cap.read()
                        if not ret:
                            st.session_state.webcam_error = f"Failed to capture frame from webcam at index {st.session_state.webcam_index}."
                            st.session_state.webcam_active = False
                            frame_placeholder.image(placeholder_image, caption="No webcam feed", use_column_width=True)
                            break
                        
                        annotated_frame, detection_count = process_frame(
                            frame, model, st.session_state.conf_threshold, st.session_state.iou_threshold
                        )
                        
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(annotated_frame_rgb, caption=f"Webcam Feed ({detection_count} detections)", use_column_width=True)
                        
                        if st.session_state.auto_adjust:
                            new_conf, new_iou, message = auto_adjust_thresholds(
                                detection_count,
                                st.session_state.conf_threshold,
                                st.session_state.iou_threshold
                            )
                            st.session_state.conf_threshold = new_conf
                            st.session_state.iou_threshold = new_iou
                            st.session_state.adjustment_message = message
                        
                        time.sleep(0.05)  # ~20 FPS for smoother performance
                
                except Exception as e:
                    st.session_state.webcam_error = f"Error processing webcam feed: {str(e)}"
                    logger.error(f"Webcam processing failed: {e}")
                    st.session_state.webcam_active = False
                    frame_placeholder.image(placeholder_image, caption="No webcam feed", use_column_width=True)
                finally:
                    cap.release()
        
        if st.session_state.webcam_error:
            st.error(f"Webcam Error: {st.session_state.webcam_error}")
            st.markdown("""
                **Troubleshooting Tips**:
                - Ensure your webcam is connected and not in use by another application.
                - Try selecting a different webcam index from the sidebar.
                - Check system permissions for camera access (Settings > Privacy > Camera).
                - Verify OpenCV is installed (`pip install opencv-python`).
                - Update webcam drivers or try a different USB port.
                - Run a test script to debug: `python -c "import cv2; cap=cv2.VideoCapture(0); print('Open' if cap.isOpened() else 'Closed'); cap.release()"`
            """)
        else:
            frame_placeholder.image(placeholder_image, caption="No webcam feed", use_column_width=True)
    
    with tab2:
        st.header("Image Upload")
        uploaded_file = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                st.write("Processing image...")
                detected_image, detection_count = process_image(image, model, st.session_state.conf_threshold, st.session_state.iou_threshold)
                st.image(detected_image, caption=f"Detected Humans ({detection_count} detections)", use_column_width=True)
                
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
                logger.error(f"Image processing failed: {e}")

if __name__ == "__main__":
    main()
