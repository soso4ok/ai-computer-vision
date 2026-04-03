"""
REST API Module

FastAPI-based REST API for accessing detection data, statistics, and real-time streams.
"""

from typing import List, Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import time


# Pydantic models for API responses
class HealthResponse(BaseModel):
    status: str
    uptime: float
    version: str = "1.0.0"


class ZoneInfo(BaseModel):
    name: str
    description: str
    points: List[List[float]]


class ZoneStats(BaseModel):
    zone_name: str
    total_visitors: int
    current_visitors: int
    total_dwell_time: float
    avg_dwell_time: float
    max_dwell_time: float
    min_dwell_time: float


class DwellRecord(BaseModel):
    track_id: int
    zone_name: str
    duration: float
    is_active: bool


class DetectionStats(BaseModel):
    total_detections: int
    active_tracks: int
    fps: float
    processing_time_ms: float


class Detection(BaseModel):
    track_id: int
    bbox: List[float]
    zone: Optional[str] = None


# API Application
def create_api(
    get_stats_callback,
    get_zones_callback,
    get_zone_dwell_callback,
    get_detections_callback,
    cors_origins: List[str] = ["*"]
) -> FastAPI:
    """
    Create the FastAPI application.
    
    Args:
        get_stats_callback: Callback to get current stats
        get_zones_callback: Callback to get zone info
        get_zone_dwell_callback: Callback to get zone dwell data
        get_detections_callback: Callback to get current detections
        cors_origins: Allowed CORS origins
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="AI Computer Vision API",
        description="Customer detection and dwell-time tracking API",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store start time
    app.state.start_time = time.time()
    
    # WebSocket connections
    app.state.ws_connections: List[WebSocket] = []
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            uptime=time.time() - app.state.start_time
        )
    
    @app.get("/stats", response_model=DetectionStats)
    async def get_stats():
        """Get current detection statistics."""
        stats = get_stats_callback()
        return DetectionStats(**stats)
    
    @app.get("/zones", response_model=List[ZoneInfo])
    async def list_zones():
        """List all configured monitoring zones."""
        zones = get_zones_callback()
        return [ZoneInfo(**z) for z in zones]
    
    @app.get("/zones/{zone_name}", response_model=ZoneInfo)
    async def get_zone(zone_name: str):
        """Get information about a specific zone."""
        zones = get_zones_callback()
        for z in zones:
            if z["name"] == zone_name:
                return ZoneInfo(**z)
        raise HTTPException(status_code=404, detail="Zone not found")
    
    @app.get("/zones/{zone_name}/dwell", response_model=ZoneStats)
    async def get_zone_dwell(zone_name: str):
        """Get dwell-time statistics for a zone."""
        stats = get_zone_dwell_callback(zone_name)
        if stats is None:
            raise HTTPException(status_code=404, detail="Zone not found")
        return ZoneStats(**stats)
    
    @app.get("/zones/{zone_name}/visitors", response_model=List[DwellRecord])
    async def get_zone_visitors(zone_name: str, active_only: bool = True):
        """Get current visitors in a zone."""
        stats = get_zone_dwell_callback(zone_name, include_visitors=True)
        if stats is None:
            raise HTTPException(status_code=404, detail="Zone not found")
        
        visitors = stats.get("visitors", [])
        if active_only:
            visitors = [v for v in visitors if v.get("is_active", False)]
        
        return [DwellRecord(**v) for v in visitors]
    
    @app.get("/detections", response_model=List[Detection])
    async def get_current_detections():
        """Get current tracked detections."""
        detections = get_detections_callback()
        return [Detection(**d) for d in detections]
    
    @app.websocket("/detections/stream")
    async def detection_stream(websocket: WebSocket):
        """WebSocket endpoint for real-time detection stream."""
        await websocket.accept()
        app.state.ws_connections.append(websocket)
        
        try:
            while True:
                # Send detection updates
                detections = get_detections_callback()
                stats = get_stats_callback()
                
                await websocket.send_json({
                    "type": "update",
                    "timestamp": time.time(),
                    "detections": detections,
                    "stats": stats
                })
                
                await asyncio.sleep(0.1)  # 10 updates per second
                
        except WebSocketDisconnect:
            app.state.ws_connections.remove(websocket)
    
    return app


class APIServer:
    """Wrapper class for running the API server."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        cors_origins: List[str] = ["*"]
    ):
        self.host = host
        self.port = port
        self.cors_origins = cors_origins
        self.app: Optional[FastAPI] = None
        self._stats = {
            "total_detections": 0,
            "active_tracks": 0,
            "fps": 0.0,
            "processing_time_ms": 0.0
        }
        self._zones = []
        self._zone_dwells = {}
        self._detections = []
        
    def setup(
        self,
        zone_manager,
        dwell_tracker
    ):
        """Set up the API with callbacks to main components."""
        self._zone_manager = zone_manager
        self._dwell_tracker = dwell_tracker
        
        def get_stats():
            return self._stats
        
        def get_zones():
            return [
                {
                    "name": z.name,
                    "description": z.description,
                    "points": [list(p) for p in z.points]
                }
                for z in self._zone_manager.list_zones()
            ]
        
        def get_zone_dwell(zone_name, include_visitors=False):
            stats = self._dwell_tracker.get_zone_stats(zone_name)
            if stats is None:
                return None
            
            result = {
                "zone_name": stats.zone_name,
                "total_visitors": stats.total_visitors,
                "current_visitors": stats.current_visitors,
                "total_dwell_time": stats.total_dwell_time,
                "avg_dwell_time": stats.avg_dwell_time,
                "max_dwell_time": stats.max_dwell_time,
                "min_dwell_time": stats.min_dwell_time if stats.min_dwell_time != float('inf') else 0.0
            }
            
            if include_visitors:
                active = self._dwell_tracker.get_active_dwells(zone_name)
                result["visitors"] = [
                    {
                        "track_id": r.track_id,
                        "zone_name": r.zone_name,
                        "duration": r.duration,
                        "is_active": r.is_active
                    }
                    for r in active
                ]
            
            return result
        
        def get_detections():
            return self._detections
        
        self.app = create_api(
            get_stats_callback=get_stats,
            get_zones_callback=get_zones,
            get_zone_dwell_callback=get_zone_dwell,
            get_detections_callback=get_detections,
            cors_origins=self.cors_origins
        )
        
    def update_stats(
        self,
        total_detections: int,
        active_tracks: int,
        fps: float,
        processing_time_ms: float
    ):
        """Update statistics."""
        self._stats = {
            "total_detections": total_detections,
            "active_tracks": active_tracks,
            "fps": fps,
            "processing_time_ms": processing_time_ms
        }
        
    def update_detections(self, detections: List[dict]):
        """Update current detections."""
        self._detections = detections
        
    async def run(self):
        """Run the API server."""
        import uvicorn
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
