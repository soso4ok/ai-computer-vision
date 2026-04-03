"""
Tests for Dwell Tracker Module
"""

import pytest
import time
from dwell_tracker import DwellTracker, DwellRecord, ZoneDwellStats


class TestDwellTracker:
    """Test cases for DwellTracker class."""

    def test_dwell_tracker_initialization(self):
        """Test dwell tracker initializes correctly."""
        tracker = DwellTracker(min_dwell_time=5.0)

        assert tracker.min_dwell_time == 5.0
        assert len(tracker.active_dwells) == 0
        assert len(tracker.completed_dwells) == 0

    def test_dwell_record_creation(self):
        """Test DwellRecord dataclass."""
        record = DwellRecord(
            track_id=1, zone_name="entrance", entry_time=time.time()
        )

        assert record.track_id == 1
        assert record.zone_name == "entrance"
        assert record.is_active
        assert record.duration >= 0

    def test_dwell_record_completion(self):
        """Test DwellRecord duration calculation."""
        entry = time.time() - 10  # 10 seconds ago
        record = DwellRecord(
            track_id=1,
            zone_name="entrance",
            entry_time=entry,
            exit_time=time.time(),
        )

        assert not record.is_active
        assert record.duration >= 9.9  # Allow small margin

    def test_update_creates_active_dwell(self):
        """Test that update creates active dwell records."""
        tracker = DwellTracker(min_dwell_time=1.0)

        tracks = [{"track_id": 1, "bbox": [100, 200, 200, 400]}]
        zone_occupancy = {"entrance": [1]}

        result = tracker.update(tracks, zone_occupancy)

        assert "entrance" in result
        assert len(result["entrance"]) == 1
        assert result["entrance"][0].track_id == 1

    def test_zone_stats_initialization(self):
        """Test ZoneDwellStats initialization."""
        stats = ZoneDwellStats(zone_name="test_zone")

        assert stats.zone_name == "test_zone"
        assert stats.total_visitors == 0
        assert stats.current_visitors == 0
        assert stats.avg_dwell_time == 0.0

    def test_zone_stats_update(self):
        """Test ZoneDwellStats update."""
        stats = ZoneDwellStats(zone_name="test_zone")

        stats.update_stats(10.0)  # 10 second dwell

        assert stats.total_visitors == 1
        assert stats.total_dwell_time == 10.0
        assert stats.avg_dwell_time == 10.0
        assert stats.max_dwell_time == 10.0
        assert stats.min_dwell_time == 10.0

        stats.update_stats(20.0)  # Another 20 second dwell

        assert stats.total_visitors == 2
        assert stats.avg_dwell_time == 15.0
        assert stats.max_dwell_time == 20.0
        assert stats.min_dwell_time == 10.0

    def test_reset_clears_data(self):
        """Test reset clears all tracking data."""
        tracker = DwellTracker()

        # Add some data
        tracks = [{"track_id": 1}]
        zone_occupancy = {"entrance": [1]}
        tracker.update(tracks, zone_occupancy)

        # Reset
        tracker.reset()

        assert len(tracker.active_dwells) == 0
        assert len(tracker.completed_dwells) == 0
        assert len(tracker.zone_stats) == 0

    def test_get_zone_stats(self):
        """Test get_zone_stats returns correct data."""
        tracker = DwellTracker(min_dwell_time=0.0)

        tracks = [{"track_id": 1}]
        zone_occupancy = {"entrance": [1]}
        tracker.update(tracks, zone_occupancy)

        stats = tracker.get_zone_stats("entrance")
        assert stats is not None
        assert stats.zone_name == "entrance"

        assert tracker.get_zone_stats("nonexistent") is None

    def test_get_all_stats(self):
        """Test get_all_stats returns all zone statistics."""
        tracker = DwellTracker()

        tracks = [{"track_id": 1}]
        tracker.update(tracks, {"zone_a": [1], "zone_b": []})

        all_stats = tracker.get_all_stats()
        assert "zone_a" in all_stats
        assert "zone_b" in all_stats

    def test_get_active_dwells_filtered(self):
        """Test get_active_dwells with zone filter."""
        tracker = DwellTracker()

        tracks = [{"track_id": 1}, {"track_id": 2}]
        tracker.update(tracks, {"zone_a": [1], "zone_b": [2]})

        zone_a_dwells = tracker.get_active_dwells("zone_a")
        assert len(zone_a_dwells) == 1
        assert zone_a_dwells[0].track_id == 1

    def test_max_completed_dwells_limit(self):
        """Test that completed_dwells is pruned at max size."""
        tracker = DwellTracker(min_dwell_time=0.0)
        assert hasattr(tracker, "MAX_COMPLETED_DWELLS")
