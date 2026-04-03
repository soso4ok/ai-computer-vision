"""
Tests for REST API Module
"""

import pytest
import time
from unittest.mock import Mock

from api import create_api, APIServer


@pytest.fixture
def mock_callbacks():
    """Create mock callback functions for the API."""
    stats = {
        "total_detections": 42,
        "active_tracks": 3,
        "fps": 25.0,
        "processing_time_ms": 33.5,
    }
    zones = [
        {
            "name": "entrance",
            "description": "Main entrance",
            "points": [[0.0, 0.6], [0.3, 0.6], [0.3, 1.0], [0.0, 1.0]],
        }
    ]
    zone_dwell = {
        "zone_name": "entrance",
        "total_visitors": 10,
        "current_visitors": 2,
        "total_dwell_time": 150.0,
        "avg_dwell_time": 15.0,
        "max_dwell_time": 45.0,
        "min_dwell_time": 5.0,
    }
    detections = [
        {"track_id": 1, "bbox": [100, 200, 300, 400], "zone": "entrance"},
        {"track_id": 2, "bbox": [400, 200, 500, 400], "zone": None},
    ]

    return {
        "get_stats": Mock(return_value=stats),
        "get_zones": Mock(return_value=zones),
        "get_zone_dwell": Mock(return_value=zone_dwell),
        "get_detections": Mock(return_value=detections),
    }


@pytest.fixture
def app(mock_callbacks):
    """Create a test FastAPI app."""
    return create_api(
        get_stats_callback=mock_callbacks["get_stats"],
        get_zones_callback=mock_callbacks["get_zones"],
        get_zone_dwell_callback=mock_callbacks["get_zone_dwell"],
        get_detections_callback=mock_callbacks["get_detections"],
    )


@pytest.fixture
def client(app):
    """Create a test client."""
    from fastapi.testclient import TestClient

    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "uptime" in data
        assert data["version"] == "1.0.0"


class TestStatsEndpoint:
    """Tests for /stats endpoint."""

    def test_get_stats(self, client):
        """Test stats endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_detections"] == 42
        assert data["active_tracks"] == 3
        assert data["fps"] == 25.0


class TestZonesEndpoints:
    """Tests for /zones endpoints."""

    def test_list_zones(self, client):
        """Test listing zones."""
        response = client.get("/zones")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "entrance"

    def test_get_zone(self, client):
        """Test getting a specific zone."""
        response = client.get("/zones/entrance")
        assert response.status_code == 200
        assert response.json()["name"] == "entrance"

    def test_get_zone_not_found(self, client):
        """Test getting a non-existent zone."""
        response = client.get("/zones/nonexistent")
        assert response.status_code == 404

    def test_get_zone_dwell(self, client):
        """Test getting dwell stats for a zone."""
        response = client.get("/zones/entrance/dwell")
        assert response.status_code == 200

        data = response.json()
        assert data["zone_name"] == "entrance"
        assert data["total_visitors"] == 10
        assert data["avg_dwell_time"] == 15.0

    def test_get_zone_dwell_not_found(self, client, mock_callbacks):
        """Test dwell stats for non-existent zone."""
        mock_callbacks["get_zone_dwell"].return_value = None
        response = client.get("/zones/nonexistent/dwell")
        assert response.status_code == 404


class TestDetectionsEndpoint:
    """Tests for /detections endpoint."""

    def test_get_detections(self, client):
        """Test getting current detections."""
        response = client.get("/detections")
        assert response.status_code == 200

        data = response.json()
        assert len(data) == 2
        assert data[0]["track_id"] == 1
        assert data[0]["zone"] == "entrance"
        assert data[1]["zone"] is None


class TestAPIServer:
    """Tests for APIServer wrapper class."""

    def test_api_server_initialization(self):
        """Test APIServer initializes with defaults."""
        server = APIServer()
        assert server.host == "0.0.0.0"
        assert server.port == 8000
        assert server.cors_origins == ["*"]

    def test_api_server_custom_cors(self):
        """Test APIServer with custom CORS origins."""
        server = APIServer(cors_origins=["http://localhost:3000"])
        assert server.cors_origins == ["http://localhost:3000"]

    def test_api_server_none_cors_defaults(self):
        """Test APIServer with None CORS defaults to wildcard."""
        server = APIServer(cors_origins=None)
        assert server.cors_origins == ["*"]

    def test_update_stats(self):
        """Test updating stats."""
        server = APIServer()
        server.update_stats(
            total_detections=100,
            active_tracks=5,
            fps=30.0,
            processing_time_ms=20.0,
        )
        assert server._stats["total_detections"] == 100
        assert server._stats["fps"] == 30.0

    def test_update_detections(self):
        """Test updating detections."""
        server = APIServer()
        dets = [{"track_id": 1, "bbox": [0, 0, 100, 100], "zone": None}]
        server.update_detections(dets)
        assert server._detections == dets


class TestCreateApiFactory:
    """Tests for create_api factory function."""

    def test_create_api_default_cors(self):
        """Test create_api with default CORS (None -> wildcard)."""
        app = create_api(
            get_stats_callback=Mock(),
            get_zones_callback=Mock(),
            get_zone_dwell_callback=Mock(),
            get_detections_callback=Mock(),
        )
        assert app is not None

    def test_create_api_custom_cors(self):
        """Test create_api with custom CORS."""
        app = create_api(
            get_stats_callback=Mock(),
            get_zones_callback=Mock(),
            get_zone_dwell_callback=Mock(),
            get_detections_callback=Mock(),
            cors_origins=["http://example.com"],
        )
        assert app is not None
