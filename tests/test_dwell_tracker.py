"""
Tests for Dwell Tracker Module
"""

import pytest
import time
from unittest.mock import patch


class TestDwellTracker:
    """Test cases for DwellTracker class."""
    
    def test_dwell_tracker_initialization(self):
        """Test dwell tracker initializes correctly."""
        import sys
        sys.path.insert(0, 'src')
        from dwell_tracker import DwellTracker
        
        tracker = DwellTracker(min_dwell_time=5.0)
        
        assert tracker.min_dwell_time == 5.0
        assert len(tracker.active_dwells) == 0
        assert len(tracker.completed_dwells) == 0
    
    def test_dwell_record_creation(self):
        """Test DwellRecord dataclass."""
        import sys
        sys.path.insert(0, 'src')
        from dwell_tracker import DwellRecord
        
        record = DwellRecord(
            track_id=1,
            zone_name="entrance",
            entry_time=time.time()
        )
        
        assert record.track_id == 1
        assert record.zone_name == "entrance"
        assert record.is_active
        assert record.duration >= 0
    
    def test_dwell_record_completion(self):
        """Test DwellRecord duration calculation."""
        import sys
        sys.path.insert(0, 'src')
        from dwell_tracker import DwellRecord
        
        entry = time.time() - 10  # 10 seconds ago
        record = DwellRecord(
            track_id=1,
            zone_name="entrance",
            entry_time=entry,
            exit_time=time.time()
        )
        
        assert not record.is_active
        assert record.duration >= 9.9  # Allow small margin
    
    def test_update_creates_active_dwell(self):
        """Test that update creates active dwell records."""
        import sys
        sys.path.insert(0, 'src')
        from dwell_tracker import DwellTracker
        
        tracker = DwellTracker(min_dwell_time=1.0)
        
        tracks = [{"track_id": 1, "bbox": [100, 200, 200, 400]}]
        zone_occupancy = {"entrance": [1]}
        
        result = tracker.update(tracks, zone_occupancy)
        
        assert "entrance" in result
        assert len(result["entrance"]) == 1
        assert result["entrance"][0].track_id == 1
    
    def test_zone_stats_initialization(self):
        """Test ZoneDwellStats initialization."""
        import sys
        sys.path.insert(0, 'src')
        from dwell_tracker import ZoneDwellStats
        
        stats = ZoneDwellStats(zone_name="test_zone")
        
        assert stats.zone_name == "test_zone"
        assert stats.total_visitors == 0
        assert stats.current_visitors == 0
        assert stats.avg_dwell_time == 0.0
    
    def test_zone_stats_update(self):
        """Test ZoneDwellStats update."""
        import sys
        sys.path.insert(0, 'src')
        from dwell_tracker import ZoneDwellStats
        
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
        import sys
        sys.path.insert(0, 'src')
        from dwell_tracker import DwellTracker
        
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


class TestZoneManager:
    """Test cases for ZoneManager class."""
    
    def test_zone_manager_initialization(self):
        """Test zone manager initializes correctly."""
        import sys
        sys.path.insert(0, 'src')
        from zone_manager import ZoneManager
        
        zones_config = [
            {
                "name": "entrance",
                "description": "Main entrance",
                "points": [[0, 0.5], [0.3, 0.5], [0.3, 1.0], [0, 1.0]]
            }
        ]
        
        manager = ZoneManager(zones_config)
        
        assert len(manager.zones) == 1
        assert "entrance" in manager.zones
    
    def test_add_zone(self):
        """Test adding a zone."""
        import sys
        sys.path.insert(0, 'src')
        from zone_manager import ZoneManager
        
        manager = ZoneManager()
        
        zone = manager.add_zone(
            name="test",
            description="Test zone",
            points=[[0, 0], [1, 0], [1, 1], [0, 1]]
        )
        
        assert zone.name == "test"
        assert len(manager.zones) == 1
    
    def test_remove_zone(self):
        """Test removing a zone."""
        import sys
        sys.path.insert(0, 'src')
        from zone_manager import ZoneManager
        
        manager = ZoneManager()
        manager.add_zone("test", "Test", [[0, 0], [1, 0], [1, 1], [0, 1]])
        
        result = manager.remove_zone("test")
        
        assert result is True
        assert "test" not in manager.zones
    
    def test_point_in_zone(self):
        """Test point in zone detection."""
        import sys
        sys.path.insert(0, 'src')
        from zone_manager import ZoneManager
        
        manager = ZoneManager()
        manager.add_zone(
            name="test",
            description="Full frame zone",
            points=[[0, 0], [1, 0], [1, 1], [0, 1]]
        )
        
        # Point in center should be inside
        result = manager.check_point_in_zone(
            point=(500, 360),
            zone_name="test",
            frame_size=(1000, 720)
        )
        
        assert result is True
    
    def test_get_zone_occupancy(self):
        """Test zone occupancy detection."""
        import sys
        sys.path.insert(0, 'src')
        from zone_manager import ZoneManager
        
        manager = ZoneManager()
        manager.add_zone(
            name="full",
            description="Full frame",
            points=[[0, 0], [1, 0], [1, 1], [0, 1]]
        )
        
        tracks = [
            {"track_id": 1, "bbox": [400, 300, 600, 500]}
        ]
        
        occupancy = manager.get_zone_occupancy(tracks, frame_size=(1000, 720))
        
        assert "full" in occupancy
        assert 1 in occupancy["full"]
