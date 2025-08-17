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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RTC configuration
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Cache model loading
@st.cache_resource
def load_model(model_path="yolo11n_human_detection_final.pt"):
    try:
        model = YOLO(model_path)
        logger.info(f"Model loaded successfully. Class names: {model.names}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# Video processor class
class VideoProcessor:
    def __init__(self, model, conf_threshold, iou_threshold):
        self.model = model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        logger.info(f"VideoProcessor initialized with conf_threshold={conf_threshold}, iou_threshold={iou_threshold}")

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            results = self.model.predict(
                img,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Get boxes and class IDs
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Draw custom boxes and labels
            annotated_img = img.copy()
            for box, class_id, conf in zip(boxes, class_ids, confidences):
                x1, y1, x2, y2 = map(int, box)
                label = "human"  # Force label to "human"
                # Draw green rectangle
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
                # Draw label text
                cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")
        except Exception as e:
            logger.error(f"Error processing webcam frame: {e}")
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
            verbose=False
        )
        
        # Get boxes and class IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        
        # Draw custom boxes and labels
        annotated_img = img_array.copy()
        for box, class_id, conf in zip(boxes, class_ids, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = "human"  # Force label to "human"
            # Draw green rectangle
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            # Draw label text
            cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(annotated_img), len(results[0].boxes)
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise

def main():
    st.set_page_config(page_title="YOLOv11 Human Detection", layout="wide")
    
    st.title("Real-Time Human Detection with YOLOv11")
    st.markdown("""
        This app performs real-time human detection using a trained YOLOv11 model.
        Use your webcam for live detection or upload an image for static analysis.
    """)
    
    # Fixed thresholds
    conf_threshold = 0.10
    iou_threshold = 0.45
    model_path = "yolo11n_human_detection_final.pt"
    
    # Load model
    model = None
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            st.success(f"Model loaded successfully! Class names: {model.names}")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            logger.error(f"Model loading failed: {e}")
            return
    else:
        st.error("Model file not found.")
        return
    
    # Tabs for webcam and image upload
    tab1, tab2 = st.tabs(["Webcam Detection", "Image Upload"])
    
    with tab1:
        st.header("Webcam Detection")
        st.write("Click 'Start' to begin real-time human detection using your webcam.")
        
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
                st.info("Webcam detection is active.")
            else:
                st.warning("Webcam stream not active. Click 'Start' or check your webcam.")
        
        except Exception as e:
            st.error(f"Failed to initialize webcam: {str(e)}")
            logger.error(f"WebRTC initialization failed: {e}")
    
    with tab2:
        st.header("Image Upload")
        uploaded_file = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                st.write("Processing image...")
                detected_image, detection_count = process_image(image, model, conf_threshold, iou_threshold)
                st.image(detected_image, caption=f"Detected Humans ({detection_count} detections)", use_column_width=True)
                
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
