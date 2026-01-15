# Models Directory

This directory should contain the YOLO model file trained for license plate detection.

1. Train your model using YOLOv8/v11 on Roboflow or locally.
2. Export the weights as `best.pt`.
3. Place `best.pt` in this directory.

If `best.pt` is not found, the application will attempt to use `yolov8n.pt` (standard YOLO model) for demonstration purposes.
