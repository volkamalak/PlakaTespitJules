import os
import time
import cv2
from ultralytics import YOLO
from PIL import Image

class LicensePlateDetector:
    def __init__(self, model_path=None):
        """
        Initialize the detector with the given model path.
        If the model path doesn't exist, falls back to yolov8n.pt.
        """
        if model_path is None:
            # Default location
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best.pt')

        self.model_path = model_path

        if not os.path.exists(self.model_path):
            print(f"Custom model not found at {self.model_path}. Using standard yolov8n.pt for demonstration.")
            self.model_name = 'yolov8n.pt'
        else:
            print(f"Loading model from {self.model_path}...")
            self.model_name = self.model_path

        # Load the model
        try:
            self.model = YOLO(self.model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def detect(self, image_path):
        """
        Detects license plates in the given image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            tuple: (processed_image (PIL.Image), coordinates (list of dicts), time_taken (float))
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Start timer
        start_time = time.time()

        # Run inference
        # conf=0.25 is standard confidence threshold
        results = self.model(image_path, conf=0.25)

        end_time = time.time()
        time_taken = end_time - start_time

        # Process results
        coordinates = []

        # We assume one image is passed, so we take the first result
        result = results[0]

        # Draw bounding boxes
        # result.plot() returns a BGR numpy array with boxes drawn
        plotted_image_bgr = result.plot()

        # Convert BGR to RGB for PIL/Tkinter
        plotted_image_rgb = cv2.cvtColor(plotted_image_bgr, cv2.COLOR_BGR2RGB)
        processed_image = Image.fromarray(plotted_image_rgb)

        # Extract coordinates
        for box in result.boxes:
            # box.xyxy contains [x1, y1, x2, y2]
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [int(c) for c in coords]

            # Get confidence and class
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = result.names[cls]

            coordinates.append({
                'label': class_name,
                'confidence': conf,
                'bbox': (x1, y1, x2, y2)
            })

        return processed_image, coordinates, time_taken

if __name__ == "__main__":
    # Simple test if run directly
    detector = LicensePlateDetector()
    print("Detector initialized successfully.")
