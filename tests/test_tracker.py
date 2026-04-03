"""
Tests for Object Tracker Module
"""

import pytest
import numpy as np
from tracker import ObjectTracker, KalmanBoxTracker, iou


class TestObjectTracker:
    """Test cases for ObjectTracker class."""

    def test_tracker_initialization(self):
        """Test tracker initializes with correct parameters."""
        tracker = ObjectTracker(max_age=30, min_hits=3, iou_threshold=0.3)

        assert tracker.max_age == 30
        assert tracker.min_hits == 3
        assert tracker.iou_threshold == 0.3
        assert tracker.frame_count == 0
        assert len(tracker.trackers) == 0

    def test_tracker_update_empty(self):
        """Test tracker update with empty detections."""
        tracker = ObjectTracker()
        tracks = tracker.update([])

        assert isinstance(tracks, list)
        assert len(tracks) == 0

    def test_tracker_update_single_detection(self):
        """Test tracker update with single detection."""
        tracker = ObjectTracker(min_hits=1)

        detection = {"bbox": [100, 200, 200, 400]}

        # First update creates track
        tracks = tracker.update([detection])

        assert len(tracker.trackers) == 1

    def test_tracker_assigns_unique_ids(self):
        """Test that tracker assigns unique IDs."""
        # Reset counter
        KalmanBoxTracker.count = 0

        tracker = ObjectTracker(min_hits=1)

        detections = [
            {"bbox": [100, 200, 200, 400]},
            {"bbox": [300, 200, 400, 400]},
        ]

        # Update with multiple detections
        for _ in range(3):  # Multiple frames to build tracks
            tracks = tracker.update(detections)

        if len(tracks) >= 2:
            ids = [t["track_id"] for t in tracks]
            assert len(ids) == len(set(ids)), "Track IDs should be unique"

    def test_tracker_reset(self):
        """Test tracker reset functionality."""
        tracker = ObjectTracker()
        tracker.update([{"bbox": [100, 200, 200, 400]}])

        assert len(tracker.trackers) > 0

        tracker.reset()

        assert len(tracker.trackers) == 0
        assert tracker.frame_count == 0


class TestIOU:
    """Test IOU calculation."""

    def test_iou_identical_boxes(self):
        """Test IOU of identical boxes is 1."""
        bbox = np.array([100, 200, 200, 400])

        result = iou(bbox, bbox)

        assert result == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        """Test IOU of non-overlapping boxes is 0."""
        bbox1 = np.array([0, 0, 100, 100])
        bbox2 = np.array([200, 200, 300, 300])

        result = iou(bbox1, bbox2)

        assert result == 0.0

    def test_iou_partial_overlap(self):
        """Test IOU of partially overlapping boxes."""
        bbox1 = np.array([0, 0, 100, 100])
        bbox2 = np.array([50, 50, 150, 150])

        result = iou(bbox1, bbox2)

        assert 0 < result < 1
        # Expected: intersection = 50*50=2500, union = 10000+10000-2500=17500
        # IOU = 2500/17500 ≈ 0.143
        assert result == pytest.approx(0.143, rel=0.1)


class TestKalmanBoxTracker:
    """Test Kalman filter tracker."""

    def test_kalman_tracker_predict(self):
        """Test Kalman tracker prediction."""
        KalmanBoxTracker.count = 0

        bbox = np.array([100, 200, 200, 400])
        tracker = KalmanBoxTracker(bbox)

        prediction = tracker.predict()

        assert len(prediction) == 4

    def test_kalman_tracker_update(self):
        """Test Kalman tracker update."""
        KalmanBoxTracker.count = 0

        bbox = np.array([100, 200, 200, 400])
        tracker = KalmanBoxTracker(bbox)

        new_bbox = np.array([105, 205, 205, 405])
        tracker.update(new_bbox)

        assert tracker.hits == 1
        assert tracker.time_since_update == 0
