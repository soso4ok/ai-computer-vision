"""
Zone Management Module

Handles configuration and monitoring of detection zones.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class Zone:
    """Represents a monitoring zone."""
    name: str
    description: str
    points: List[Tuple[float, float]]  # Normalized coordinates (0-1)
    
    def to_absolute(self, width: int, height: int) -> np.ndarray:
        """Convert normalized points to absolute pixel coordinates."""
        return np.array([
            [int(p[0] * width), int(p[1] * height)]
            for p in self.points
        ], dtype=np.int32)


class ZoneManager:
    """Manages monitoring zones and checks occupancy."""
    
    def __init__(self, zones_config: Optional[List[dict]] = None):
        """
        Initialize the zone manager.
        
        Args:
            zones_config: List of zone configuration dictionaries
        """
        self.zones: Dict[str, Zone] = {}
        
        if zones_config:
            for zone_cfg in zones_config:
                self.add_zone(
                    name=zone_cfg["name"],
                    description=zone_cfg.get("description", ""),
                    points=zone_cfg["points"]
                )
    
    def add_zone(
        self,
        name: str,
        description: str,
        points: List[List[float]]
    ) -> Zone:
        """
        Add a new monitoring zone.
        
        Args:
            name: Unique zone identifier
            description: Human-readable description
            points: List of [x, y] normalized coordinates
            
        Returns:
            Created Zone object
        """
        zone = Zone(
            name=name,
            description=description,
            points=[tuple(p) for p in points]
        )
        self.zones[name] = zone
        return zone
    
    def remove_zone(self, name: str) -> bool:
        """Remove a zone by name."""
        if name in self.zones:
            del self.zones[name]
            return True
        return False
    
    def get_zone(self, name: str) -> Optional[Zone]:
        """Get a zone by name."""
        return self.zones.get(name)
    
    def list_zones(self) -> List[Zone]:
        """List all configured zones."""
        return list(self.zones.values())
    
    def check_point_in_zone(
        self,
        point: Tuple[float, float],
        zone_name: str,
        frame_size: Tuple[int, int]
    ) -> bool:
        """
        Check if a point is inside a zone.
        
        Args:
            point: (x, y) pixel coordinates
            zone_name: Name of zone to check
            frame_size: (width, height) of the frame
            
        Returns:
            True if point is inside zone
        """
        zone = self.zones.get(zone_name)
        if not zone:
            return False
            
        width, height = frame_size
        polygon = zone.to_absolute(width, height)
        
        return self._point_in_polygon(point, polygon)
    
    def check_bbox_in_zone(
        self,
        bbox: List[float],
        zone_name: str,
        frame_size: Tuple[int, int],
        method: str = "center"
    ) -> bool:
        """
        Check if a bounding box is inside a zone.
        
        Args:
            bbox: [x1, y1, x2, y2] bounding box
            zone_name: Name of zone to check
            frame_size: (width, height) of the frame
            method: Detection method - 'center', 'bottom_center', 'any_corner'
            
        Returns:
            True if bbox is considered inside zone
        """
        x1, y1, x2, y2 = bbox
        
        if method == "center":
            point = ((x1 + x2) / 2, (y1 + y2) / 2)
        elif method == "bottom_center":
            point = ((x1 + x2) / 2, y2)
        elif method == "any_corner":
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            return any(
                self.check_point_in_zone(c, zone_name, frame_size)
                for c in corners
            )
        else:
            point = ((x1 + x2) / 2, (y1 + y2) / 2)
            
        return self.check_point_in_zone(point, zone_name, frame_size)
    
    def get_zone_occupancy(
        self,
        tracks: List[dict],
        frame_size: Tuple[int, int],
        method: str = "bottom_center"
    ) -> Dict[str, List[int]]:
        """
        Determine which tracks are in which zones.
        
        Args:
            tracks: List of track dicts with 'track_id' and 'bbox'
            frame_size: (width, height) of the frame
            method: Zone detection method
            
        Returns:
            Dict mapping zone names to lists of track IDs
        """
        occupancy: Dict[str, List[int]] = {name: [] for name in self.zones}
        
        for track in tracks:
            track_id = track["track_id"]
            bbox = track["bbox"]
            
            for zone_name in self.zones:
                if self.check_bbox_in_zone(bbox, zone_name, frame_size, method):
                    occupancy[zone_name].append(track_id)
                    
        return occupancy
    
    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: np.ndarray
    ) -> bool:
        """Check if point is inside polygon using ray casting algorithm."""
        x, y = point
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
            
        return inside
    
    def draw_zones(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        fill_alpha: float = 0.2
    ) -> np.ndarray:
        """
        Draw zones on a frame.
        
        Args:
            frame: BGR image
            color: BGR color tuple
            thickness: Line thickness
            fill_alpha: Fill transparency (0-1)
            
        Returns:
            Frame with zones drawn
        """
        import cv2
        
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        for zone in self.zones.values():
            points = zone.to_absolute(width, height)
            
            # Draw filled polygon
            cv2.fillPoly(overlay, [points], color)
            
            # Draw outline
            cv2.polylines(frame, [points], True, color, thickness)
            
            # Draw zone name
            centroid = points.mean(axis=0).astype(int)
            cv2.putText(
                frame,
                zone.name,
                tuple(centroid),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # Blend overlay
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
        
        return frame
