"""
Person Detection Module using YOLOv8

Handles loading the YOLO model and performing person detection on video frames.
"""

from typing import List, Tuple, Optional
import numpy as np
from ultralytics import YOLO
import torch


class PersonDetector:
    """YOLOv8-based person detection engine."""
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence: Detection confidence threshold
            device: Device to run inference on (auto, cpu, cuda)
        """
        self.confidence = confidence
        self.device = self._resolve_device(device)
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Detect persons in a frame.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2] bounding box
            - confidence: detection confidence score
            - class_id: class identifier (0 for person)
        """
        results = self.model(
            frame,
            conf=self.confidence,
            classes=[0],  # Only persons
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                detections.append({
                    "bbox": bbox.tolist(),
                    "confidence": conf,
                    "class_id": cls
                })
                
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[dict]]:
        """
        Detect persons in multiple frames (batch processing).
        
        Args:
            frames: List of BGR images
            
        Returns:
            List of detection lists for each frame
        """
        results = self.model(
            frames,
            conf=self.confidence,
            classes=[0],
            verbose=False
        )
        
        all_detections = []
        for result in results:
            frame_detections = []
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    frame_detections.append({
                        "bbox": bbox.tolist(),
                        "confidence": conf,
                        "class_id": cls
                    })
            all_detections.append(frame_detections)
            
        return all_detections
    
    @property
    def input_size(self) -> Tuple[int, int]:
        """Get model input size."""
        return (640, 640)
    
    def warmup(self, input_shape: Tuple[int, int, int] = (720, 1280, 3)):
        """Warm up the model with a dummy inference."""
        dummy_frame = np.zeros(input_shape, dtype=np.uint8)
        self.detect(dummy_frame)
