"""
Person Detection Module using YOLOv8

Handles loading the YOLO model and performing person detection on video frames.
Supports NCNN export for optimized ARM/Raspberry Pi inference.
"""

from typing import List, Tuple, Optional
import logging
import platform
from pathlib import Path

import numpy as np
from ultralytics import YOLO
import torch


logger = logging.getLogger(__name__)


def is_arm_platform() -> bool:
    """Check if running on ARM architecture (e.g., Raspberry Pi)."""
    machine = platform.machine().lower()
    return machine in ("aarch64", "armv7l", "armv8l", "arm64")


class PersonDetector:
    """YOLOv8-based person detection engine with Raspberry Pi support."""
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.5,
        device: str = "auto",
        use_ncnn: bool = False,
    ):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence: Detection confidence threshold
            device: Device to run inference on (auto, cpu, cuda)
            use_ncnn: Export and use NCNN format for faster ARM inference
        """
        self.confidence = confidence
        self.device = self._resolve_device(device)
        self.use_ncnn = use_ncnn or is_arm_platform()
        
        # Load or export model
        self.model = self._load_model(model_path)
        
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == "auto":
            if is_arm_platform():
                logger.info("ARM platform detected, forcing CPU device")
                return "cpu"
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self, model_path: str) -> YOLO:
        """Load YOLO model, optionally exporting to NCNN for ARM."""
        model = YOLO(model_path)
        
        if self.use_ncnn:
            ncnn_dir = Path(model_path).stem + "_ncnn_model"
            if Path(ncnn_dir).exists():
                logger.info(f"Loading pre-exported NCNN model from {ncnn_dir}")
                model = YOLO(ncnn_dir)
            else:
                logger.info("Exporting model to NCNN format for ARM inference...")
                try:
                    ncnn_path = model.export(format="ncnn")
                    logger.info(f"NCNN model exported to {ncnn_path}")
                    model = YOLO(ncnn_path)
                except Exception as e:
                    logger.warning(
                        f"NCNN export failed ({e}), falling back to PyTorch model"
                    )
                    model.to(self.device)
        else:
            model.to(self.device)
        
        return model
    
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
    
    def warmup(self, input_shape: Tuple[int, int, int] = (480, 640, 3)):
        """Warm up the model with a dummy inference."""
        dummy_frame = np.zeros(input_shape, dtype=np.uint8)
        self.detect(dummy_frame)
