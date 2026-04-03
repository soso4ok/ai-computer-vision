"""
Dwell-Time Tracking Module

Calculates and monitors how long tracked persons spend in designated zones.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import threading


@dataclass
class DwellRecord:
    """Record of a person's dwell time in a zone."""
    track_id: int
    zone_name: str
    entry_time: float
    exit_time: Optional[float] = None
    
    @property
    def duration(self) -> float:
        """Get dwell duration in seconds."""
        end = self.exit_time if self.exit_time else time.time()
        return end - self.entry_time
    
    @property
    def is_active(self) -> bool:
        """Check if person is still in zone."""
        return self.exit_time is None


@dataclass
class ZoneDwellStats:
    """Statistics for dwell time in a zone."""
    zone_name: str
    total_visitors: int = 0
    current_visitors: int = 0
    total_dwell_time: float = 0.0
    avg_dwell_time: float = 0.0
    max_dwell_time: float = 0.0
    min_dwell_time: float = float('inf')
    
    def update_stats(self, duration: float):
        """Update statistics with a completed dwell record."""
        self.total_visitors += 1
        self.total_dwell_time += duration
        self.avg_dwell_time = self.total_dwell_time / self.total_visitors
        self.max_dwell_time = max(self.max_dwell_time, duration)
        self.min_dwell_time = min(self.min_dwell_time, duration)


class DwellTracker:
    """Tracks dwell time of persons in configured zones."""
    
    def __init__(self, min_dwell_time: float = 5.0):
        """
        Initialize the dwell tracker.
        
        Args:
            min_dwell_time: Minimum seconds to count as valid dwell
        """
        self.min_dwell_time = min_dwell_time
        self.active_dwells: Dict[Tuple[int, str], DwellRecord] = {}
        self.completed_dwells: List[DwellRecord] = []
        self.zone_stats: Dict[str, ZoneDwellStats] = {}
        self._lock = threading.Lock()
        
    def update(
        self,
        tracks: List[dict],
        zone_occupancy: Dict[str, List[int]]
    ) -> Dict[str, List[DwellRecord]]:
        """
        Update dwell tracking with current track positions and zone occupancy.
        
        Args:
            tracks: List of tracked objects with track_id
            zone_occupancy: Dict mapping zone names to list of track_ids in zone
            
        Returns:
            Dict mapping zone names to active dwell records
        """
        current_time = time.time()
        track_ids = {t["track_id"] for t in tracks}
        
        with self._lock:
            # Process each zone
            for zone_name, ids_in_zone in zone_occupancy.items():
                # Initialize zone stats if needed
                if zone_name not in self.zone_stats:
                    self.zone_stats[zone_name] = ZoneDwellStats(zone_name=zone_name)
                
                ids_in_zone_set = set(ids_in_zone)
                
                # Check for new entries
                for track_id in ids_in_zone:
                    key = (track_id, zone_name)
                    if key not in self.active_dwells:
                        self.active_dwells[key] = DwellRecord(
                            track_id=track_id,
                            zone_name=zone_name,
                            entry_time=current_time
                        )
                        self.zone_stats[zone_name].current_visitors += 1
                
                # Check for exits
                keys_to_complete = []
                for key, record in self.active_dwells.items():
                    if record.zone_name != zone_name:
                        continue
                    
                    track_id = key[0]
                    
                    # Check if track left zone or disappeared
                    if track_id not in ids_in_zone_set or track_id not in track_ids:
                        record.exit_time = current_time
                        keys_to_complete.append(key)
                
                # Process completed dwells
                for key in keys_to_complete:
                    record = self.active_dwells.pop(key)
                    
                    # Only count if duration meets minimum threshold
                    if record.duration >= self.min_dwell_time:
                        self.completed_dwells.append(record)
                        self.zone_stats[record.zone_name].update_stats(record.duration)
                    
                    self.zone_stats[record.zone_name].current_visitors -= 1
            
            # Build result
            result: Dict[str, List[DwellRecord]] = {}
            for key, record in self.active_dwells.items():
                zone_name = record.zone_name
                if zone_name not in result:
                    result[zone_name] = []
                result[zone_name].append(record)
                
        return result
    
    def get_zone_stats(self, zone_name: str) -> Optional[ZoneDwellStats]:
        """Get statistics for a specific zone."""
        with self._lock:
            return self.zone_stats.get(zone_name)
    
    def get_all_stats(self) -> Dict[str, ZoneDwellStats]:
        """Get statistics for all zones."""
        with self._lock:
            return dict(self.zone_stats)
    
    def get_active_dwells(self, zone_name: Optional[str] = None) -> List[DwellRecord]:
        """Get active dwell records, optionally filtered by zone."""
        with self._lock:
            if zone_name:
                return [r for r in self.active_dwells.values() if r.zone_name == zone_name]
            return list(self.active_dwells.values())
    
    def get_completed_dwells(
        self,
        zone_name: Optional[str] = None,
        since: Optional[float] = None
    ) -> List[DwellRecord]:
        """Get completed dwell records, optionally filtered."""
        with self._lock:
            records = self.completed_dwells
            
            if zone_name:
                records = [r for r in records if r.zone_name == zone_name]
            
            if since:
                records = [r for r in records if r.exit_time and r.exit_time >= since]
                
            return records
    
    def reset(self):
        """Reset all tracking data."""
        with self._lock:
            self.active_dwells.clear()
            self.completed_dwells.clear()
            self.zone_stats.clear()
