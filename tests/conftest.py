"""
Shared test configuration and fixtures.
"""

import sys
from pathlib import Path

import pytest

# Add src directory to Python path so tests can import modules directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def zone_config():
    """Sample zone configuration for testing."""
    return [
        {
            "name": "entrance",
            "description": "Main entrance area",
            "points": [[0.0, 0.6], [0.3, 0.6], [0.3, 1.0], [0.0, 1.0]],
        },
        {
            "name": "display_area",
            "description": "Product display zone",
            "points": [[0.4, 0.3], [0.8, 0.3], [0.8, 0.7], [0.4, 0.7]],
        },
    ]


@pytest.fixture
def sample_detections():
    """Sample detections for testing."""
    return [
        {"bbox": [100, 200, 200, 400], "confidence": 0.95, "class_id": 0},
        {"bbox": [300, 200, 400, 400], "confidence": 0.88, "class_id": 0},
    ]


@pytest.fixture
def sample_tracks():
    """Sample tracks for testing."""
    return [
        {"track_id": 1, "bbox": [100, 200, 200, 400], "hits": 5, "age": 10},
        {"track_id": 2, "bbox": [300, 200, 400, 400], "hits": 3, "age": 5},
    ]
