"""
Object Tracking Module

Implements SORT-like tracking algorithm to maintain consistent IDs
for detected persons across frames.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculate Intersection over Union between two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = bbox1_area + bbox2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


class KalmanBoxTracker:
    """Kalman filter-based tracker for a single bounding box."""
    
    count = 0
    
    def __init__(self, bbox: np.ndarray):
        """Initialize tracker with initial bounding box."""
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State transition matrix
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])
        
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # Initialize state with bbox converted to [x_center, y_center, scale, ratio]
        self.kf.x[:4] = self._bbox_to_z(bbox)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
    def _bbox_to_z(self, bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [x_center, y_center, scale, ratio]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2
        y = bbox[1] + h / 2
        s = w * h
        r = w / h if h > 0 else 1
        return np.array([[x], [y], [s], [r]])
    
    def _z_to_bbox(self, z: np.ndarray) -> np.ndarray:
        """Convert [x_center, y_center, scale, ratio] to [x1, y1, x2, y2]."""
        scale = np.maximum(z[2], 1e-6)
        ratio = np.maximum(z[3], 1e-6)
        w = np.sqrt(scale * ratio)
        h = scale / w if w > 0 else 0
        return np.array([
            z[0] - w / 2,
            z[1] - h / 2,
            z[0] + w / 2,
            z[1] + h / 2
        ]).flatten()
    
    def update(self, bbox: np.ndarray):
        """Update the tracker with a new detection."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self._bbox_to_z(bbox))
        
    def predict(self) -> np.ndarray:
        """Predict the next state and return the bbox."""
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self._z_to_bbox(self.kf.x[:4]))
        return self.history[-1]
    
    def get_state(self) -> np.ndarray:
        """Return current bounding box estimate."""
        return self._z_to_bbox(self.kf.x[:4])


class ObjectTracker:
    """Multi-object tracker using Kalman filtering and Hungarian algorithm."""
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize the tracker.
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits before confirming track
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        
    def update(self, detections: List[dict]) -> List[dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dicts with 'bbox' key
            
        Returns:
            List of tracked objects with track_id assigned
        """
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
                
        # Remove invalid trackers
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        
        # Extract detection bboxes
        dets = np.array([d["bbox"] for d in detections]) if detections else np.empty((0, 4))
        
        # Match detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections(dets, trks)
        
        # Update matched trackers
        for d, t in matched:
            self.trackers[t].update(dets[d])
            
        # Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)
            
        # Build output
        ret = []
        for trk in self.trackers:
            if trk.time_since_update < 1:
                if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                    ret.append({
                        "track_id": trk.id,
                        "bbox": trk.get_state().tolist(),
                        "hits": trk.hits,
                        "age": trk.age
                    })
                    
        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update < self.max_age]
        
        return ret
    
    def _associate_detections(
        self,
        detections: np.ndarray,
        trackers: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to trackers using Hungarian algorithm."""
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(trackers)))
            
        # Build IOU cost matrix
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det, trk)
                
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(trackers)))
        
        for d, t in zip(row_ind, col_ind):
            if iou_matrix[d, t] >= self.iou_threshold:
                matched.append((d, t))
                unmatched_dets.remove(d)
                unmatched_trks.remove(t)
                
        return matched, unmatched_dets, unmatched_trks
    
    def reset(self):
        """Reset all trackers."""
        self.trackers = []
        self.frame_count = 0
        KalmanBoxTracker.count = 0
