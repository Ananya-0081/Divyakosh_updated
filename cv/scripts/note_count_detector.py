"""
scripts/note_count_detector.py

Standalone YOLO-based object detector for counting physical notes in an ROI.
This module is intended to act as a robust note-presence gate before
running MobileNetV3 classification.

Goal:
Given an ROI image, return the number of notes detected (0, 1, or 2+).

Note:
Currently, the YOLO model (`models/yolo_note_detector.pt`) and the 
'ultralytics' library are not yet present in the project. This script
serves as the interface definition and placeholder implementation.

Usage (once model is available):
    from scripts.note_count_detector import NoteCountDetector
    detector = NoteCountDetector("models/yolo_note_detector.pt")
    result = detector.detect_note_count(roi_bgr_image)
"""

import os
import cv2

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class NoteCountDetector:
    def __init__(self, model_path="models/yolo_note_detector.pt", conf_thresh=0.5):
        """
        Initialize the YOLO note detector.
        
        Args:
            model_path (str): Path to the trained YOLO checkpoint.
            conf_thresh (float): Minimum confidence threshold for a valid note detection.
        """
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.model = None

        if not ULTRALYTICS_AVAILABLE:
            print("WARNING: 'ultralytics' library is not installed. Please install it using `pip install ultralytics`.")
            return

        if not os.path.exists(self.model_path):
            print(f"WARNING: YOLO model not found at '{self.model_path}'. Inference will not work.")
            return

        # Load the YOLO model (assuming YOLOv8)
        self.model = YOLO(self.model_path)
        print(f"Loaded YOLO note detector from: {self.model_path}")

    def detect_note_count(self, roi_bgr):
        """
        Detects the number of currency notes in the given ROI.

        Args:
            roi_bgr (numpy.ndarray): The BGR image crop of the region of interest.

        Returns:
            dict: A dictionary containing:
                - "count" (int): The number of notes detected.
                - "status" (str): "EMPTY", "SINGLE", or "MULTIPLE".
                - "detections" (list): List of bounding boxes and confidences.
        """
        result_dict = {
            "count": 0,
            "status": "EMPTY",
            "detections": []
        }

        if self.model is None:
            # If the model couldn't be loaded, return EMPTY by default
            return result_dict

        # Run inference
        results = self.model.predict(source=roi_bgr, conf=self.conf_thresh, verbose=False)
        
        detections = []
        # Typically YOLO predict returns a list of results (one per image)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # box.xyxy[0] has coordinates, box.conf[0] has confidence
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })

        count = len(detections)
        if count == 0:
            status = "EMPTY"
        elif count == 1:
            status = "SINGLE"
        else:
            status = "MULTIPLE"

        result_dict["count"] = count
        result_dict["status"] = status
        result_dict["detections"] = detections

        return result_dict

if __name__ == "__main__":
    print("This is a standalone module. It can be imported and used as:")
    print("  from note_count_detector import NoteCountDetector")
    print("  detector = NoteCountDetector()")
    print("  res = detector.detect_note_count(image)")
