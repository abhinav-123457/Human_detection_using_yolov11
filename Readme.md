# Human Detection with YOLOv11

This project demonstrates how to train a YOLOv11 model for human detection using a custom dataset in Google Colab. The trained model is saved to Google Drive for future use.

## Prerequisites

- **Google Colab**: Ensure you have access to a Google Colab environment with GPU support.
- **Google Drive**: A Google Drive account to store the dataset, training outputs, and the final model.
- **Ultralytics YOLO**: The `ultralytics` package installed in the Colab environment.
- **Dataset**: A custom dataset for human detection in YOLO format, including images and annotations, with a `data.yaml` configuration file.

## Setup Instructions

1. **Mount Google Drive**:
   - Run the following code to mount your Google Drive in Colab:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Follow the authentication prompt to grant access to your Google Drive.

2. **Install Dependencies**:
   - Install the `ultralytics` package:
     ```bash
     !pip install ultralytics
     ```

3. **Prepare the Dataset**:
   - Ensure your dataset is stored in `/content/Human-detection-1/` with the following structure:
     ```
     Human-detection-1/
     ├── images/
     │   ├── train/
     │   ├── val/
     │   └── test/ (optional)
     ├── labels/
     │   ├── train/
     │   ├── val/
     │   └── test/ (optional)
     └── data.yaml
     ```
   - The `data.yaml` file should define the dataset paths and classes, e.g.:
     ```yaml
     train: /content/Human-detection-1/images/train
     val: /content/Human-detection-1/images/val
     nc: 1
     names: ['human']
     ```

## Training the Model

1. **Load the Pretrained Model**:
   - The code uses the YOLOv11 nano model (`yolo11n.pt`) as the starting point:
     ```python
     from ultralytics import YOLO
     model = YOLO("yolo11n.pt")
     ```

2. **Train the Model**:
   - Train the model with the specified parameters:
     ```python
     results = model.train(
         data="/content/Human-detection-1/data.yaml",
         epochs=100,
         imgsz=640,
         batch=16,
         patience=20,
         project="/content/drive/MyDrive/YOLO_Training",
         name="yolo11n_human_detection",
         exist_ok=True
     )
     ```
   - **Parameters**:
     - `data`: Path to the `data.yaml` file.
     - `epochs`: Number of training epochs (100).
     - `imgsz`: Image size for training (640x640 pixels).
     - `batch`: Batch size (16).
     - `patience`: Early stopping patience (stops if no improvement after 20 epochs).
     - `project`: Directory to save training outputs.
     - `name`: Name of the training run.
     - `exist_ok`: Overwrites existing project folder if `True`.

3. **Save the Model**:
   - After training, the model is explicitly saved to Google Drive:
     ```python
     model.save('/content/drive/MyDrive/YOLO_Training/yolo11n_human_detection_final.pt')
     ```

## Output

- Training outputs (logs, weights, etc.) are saved in `/content/drive/MyDrive/YOLO_Training/yolo11n_human_detection/`.
- The final trained model is saved as `/content/drive/MyDrive/YOLO_Training/yolo11n_human_detection_final.pt`.

## Usage

To use the trained model for inference:
```python
from ultralytics import YOLO
model = YOLO('/content/drive/MyDrive/YOLO_Training/yolo11n_human_detection_final.pt')
results = model.predict(source="path/to/image_or_video", save=True)
```

## Notes

- Ensure sufficient Google Drive storage for training outputs and the dataset.
- Monitor GPU usage in Colab to avoid runtime disconnection.
- Adjust `epochs`, `batch`, or `imgsz` based on your dataset size and available resources.
- For larger datasets or more complex tasks, consider using a larger YOLOv11 model (e.g., `yolo11s.pt` or `yolo11m.pt`).
```
