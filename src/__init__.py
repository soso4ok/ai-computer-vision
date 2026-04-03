"""Package initialization for AI Computer Vision Server."""

__version__ = "1.0.0"
__all__ = [
    "PersonDetector",
    "ObjectTracker",
    "DwellTracker",
    "ZoneManager",
    "APIServer"
]


def __getattr__(name: str):
    """Lazy imports to avoid loading heavy dependencies on package import."""
    if name == "PersonDetector":
        from .detector import PersonDetector
        return PersonDetector
    elif name == "ObjectTracker":
        from .tracker import ObjectTracker
        return ObjectTracker
    elif name == "DwellTracker":
        from .dwell_tracker import DwellTracker
        return DwellTracker
    elif name == "ZoneManager":
        from .zone_manager import ZoneManager
        return ZoneManager
    elif name == "APIServer":
        from .api import APIServer
        return APIServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
