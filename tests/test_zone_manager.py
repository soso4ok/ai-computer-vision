"""
Tests for Zone Manager Module
"""

import pytest
import numpy as np
from zone_manager import ZoneManager, Zone


class TestZoneManager:
    """Test cases for ZoneManager class."""

    def test_zone_manager_initialization(self, zone_config):
        """Test zone manager initializes correctly."""
        manager = ZoneManager(zone_config)

        assert len(manager.zones) == 2
        assert "entrance" in manager.zones
        assert "display_area" in manager.zones

    def test_zone_manager_empty_init(self):
        """Test zone manager initializes with no zones."""
        manager = ZoneManager()
        assert len(manager.zones) == 0

    def test_add_zone(self):
        """Test adding a zone."""
        manager = ZoneManager()

        zone = manager.add_zone(
            name="test",
            description="Test zone",
            points=[[0, 0], [1, 0], [1, 1], [0, 1]],
        )

        assert zone.name == "test"
        assert zone.description == "Test zone"
        assert len(manager.zones) == 1

    def test_remove_zone(self):
        """Test removing a zone."""
        manager = ZoneManager()
        manager.add_zone("test", "Test", [[0, 0], [1, 0], [1, 1], [0, 1]])

        result = manager.remove_zone("test")

        assert result is True
        assert "test" not in manager.zones

    def test_remove_nonexistent_zone(self):
        """Test removing a zone that doesn't exist."""
        manager = ZoneManager()
        assert manager.remove_zone("nonexistent") is False

    def test_get_zone(self):
        """Test getting a zone by name."""
        manager = ZoneManager()
        manager.add_zone("test", "Test", [[0, 0], [1, 0], [1, 1], [0, 1]])

        zone = manager.get_zone("test")
        assert zone is not None
        assert zone.name == "test"

        assert manager.get_zone("nonexistent") is None

    def test_list_zones(self, zone_config):
        """Test listing all zones."""
        manager = ZoneManager(zone_config)
        zones = manager.list_zones()
        assert len(zones) == 2

    def test_point_in_zone(self):
        """Test point in zone detection."""
        manager = ZoneManager()
        manager.add_zone(
            name="test",
            description="Full frame zone",
            points=[[0, 0], [1, 0], [1, 1], [0, 1]],
        )

        # Point in center should be inside
        result = manager.check_point_in_zone(
            point=(500, 360), zone_name="test", frame_size=(1000, 720)
        )
        assert result is True

    def test_point_outside_zone(self):
        """Test point outside zone."""
        manager = ZoneManager()
        manager.add_zone(
            name="small",
            description="Small zone",
            points=[[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]],
        )

        # Point far from zone should be outside
        result = manager.check_point_in_zone(
            point=(900, 600), zone_name="small", frame_size=(1000, 720)
        )
        assert result is False

    def test_bbox_in_zone_center_method(self):
        """Test bbox in zone with center method."""
        manager = ZoneManager()
        manager.add_zone("full", "Full", [[0, 0], [1, 0], [1, 1], [0, 1]])

        result = manager.check_bbox_in_zone(
            bbox=[400, 300, 600, 500],
            zone_name="full",
            frame_size=(1000, 720),
            method="center",
        )
        assert result is True

    def test_bbox_in_zone_bottom_center_method(self):
        """Test bbox in zone with bottom_center method."""
        manager = ZoneManager()
        manager.add_zone("full", "Full", [[0, 0], [1, 0], [1, 1], [0, 1]])

        result = manager.check_bbox_in_zone(
            bbox=[400, 300, 600, 500],
            zone_name="full",
            frame_size=(1000, 720),
            method="bottom_center",
        )
        assert result is True

    def test_get_zone_occupancy(self):
        """Test zone occupancy detection."""
        manager = ZoneManager()
        manager.add_zone("full", "Full frame", [[0, 0], [1, 0], [1, 1], [0, 1]])

        tracks = [{"track_id": 1, "bbox": [400, 300, 600, 500]}]

        occupancy = manager.get_zone_occupancy(tracks, frame_size=(1000, 720))

        assert "full" in occupancy
        assert 1 in occupancy["full"]

    def test_get_zone_occupancy_empty(self):
        """Test zone occupancy with no tracks."""
        manager = ZoneManager()
        manager.add_zone("test", "Test", [[0, 0], [1, 0], [1, 1], [0, 1]])

        occupancy = manager.get_zone_occupancy([], frame_size=(1000, 720))
        assert occupancy["test"] == []


class TestZone:
    """Test Zone dataclass."""

    def test_to_absolute(self):
        """Test converting normalized coords to absolute pixels."""
        zone = Zone(
            name="test",
            description="Test",
            points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        )

        absolute = zone.to_absolute(1000, 720)
        assert absolute.shape == (4, 2)
        assert absolute[0].tolist() == [0, 0]
        assert absolute[1].tolist() == [1000, 0]
        assert absolute[2].tolist() == [1000, 720]
