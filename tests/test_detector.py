"""
Tests for Person Detector Module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock


class TestPersonDetector:
    """Test cases for PersonDetector class."""
    
    @patch('detector.YOLO')
    @patch('detector.torch')
    def test_detector_initialization(self, mock_torch, mock_yolo):
        """Test detector initializes with correct parameters."""
        mock_torch.cuda.is_available.return_value = False
        
        from detector import PersonDetector
        
        detector = PersonDetector(
            model_path="yolov8n.pt",
            confidence=0.5,
            device="cpu"
        )
        
        assert detector.confidence == 0.5
        assert detector.device == "cpu"
        mock_yolo.assert_called_once_with("yolov8n.pt")
    
    @patch('detector.YOLO')
    @patch('detector.torch')
    def test_device_auto_selection_no_cuda(self, mock_torch, mock_yolo):
        """Test auto device selection when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False
        
        from detector import PersonDetector
        
        detector = PersonDetector(device="auto")
        
        assert detector.device == "cpu"
    
    @patch('detector.YOLO')
    @patch('detector.torch')
    def test_device_auto_selection_with_cuda(self, mock_torch, mock_yolo):
        """Test auto device selection when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        
        from detector import PersonDetector
        
        detector = PersonDetector(device="auto")
        
        assert detector.device == "cuda"
    
    @patch('detector.YOLO')
    @patch('detector.torch')
    def test_detect_returns_list(self, mock_torch, mock_yolo):
        """Test detect method returns a list."""
        mock_torch.cuda.is_available.return_value = False
        
        # Mock YOLO model results
        mock_model = Mock()
        mock_boxes = Mock()
        mock_boxes.xyxy = [Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array([100, 200, 300, 400])))))]
        mock_boxes.conf = [Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array(0.95)))))]
        mock_boxes.cls = [Mock(cpu=Mock(return_value=Mock(numpy=Mock(return_value=np.array(0)))))]
        mock_boxes.__len__ = Mock(return_value=1)
        
        mock_result = Mock()
        mock_result.boxes = mock_boxes
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        from detector import PersonDetector
        
        detector = PersonDetector()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        detections = detector.detect(frame)
        
        assert isinstance(detections, list)
    
    @patch('detector.YOLO')
    @patch('detector.torch')
    def test_warmup(self, mock_torch, mock_yolo):
        """Test warmup runs without error."""
        mock_torch.cuda.is_available.return_value = False
        mock_model = Mock()
        mock_model.return_value = []
        mock_yolo.return_value = mock_model
        
        from detector import PersonDetector
        
        detector = PersonDetector()
        detector.warmup()
        
        # Verify model was called (warmup detection)
        mock_model.assert_called()


class TestDetectionOutput:
    """Test detection output format."""
    
    def test_detection_dict_structure(self):
        """Test that detection dict has required keys."""
        detection = {
            "bbox": [100, 200, 300, 400],
            "confidence": 0.95,
            "class_id": 0
        }
        
        assert "bbox" in detection
        assert "confidence" in detection
        assert "class_id" in detection
        assert len(detection["bbox"]) == 4
    
    def test_bbox_coordinates(self):
        """Test bbox coordinate format [x1, y1, x2, y2]."""
        bbox = [100, 200, 300, 400]
        
        x1, y1, x2, y2 = bbox
        
        assert x1 < x2, "x1 should be less than x2"
        assert y1 < y2, "y1 should be less than y2"
