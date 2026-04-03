"""
AI Computer Vision Server - Main Entry Point

Processes video input for customer detection and dwell-time tracking.
Supports USB webcams and Raspberry Pi Camera Module (via picamera2).
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from threading import Thread, Event

from detector import PersonDetector, is_arm_platform
from tracker import ObjectTracker
from dwell_tracker import DwellTracker
from zone_manager import ZoneManager
from api import APIServer


class PiCameraSource:
    """Wrapper for Raspberry Pi Camera Module via picamera2."""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 15):
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise RuntimeError(
                "picamera2 is not installed. Install with: "
                "sudo apt install -y python3-picamera2"
            )
        
        self.picam = Picamera2()
        config = self.picam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self.picam.configure(config)
        self.picam.start()
        self._opened = True
    
    def isOpened(self) -> bool:
        return self._opened
    
    def read(self):
        """Read a frame from the Pi camera. Returns (success, frame_bgr)."""
        try:
            frame_rgb = self.picam.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return True, frame_bgr
        except Exception:
            return False, None
    
    def release(self):
        if self._opened:
            self.picam.stop()
            self._opened = False
    
    def set(self, prop, value):
        """Compatibility stub for cv2.VideoCapture.set()."""
        pass
    
    def get(self, prop):
        """Compatibility stub for cv2.VideoCapture.get()."""
        return 0


class VisionServer:
    """Main application server for AI computer vision."""
    
    def __init__(self, config_path: str):
        """Initialize the vision server with configuration."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AI Computer Vision Server")
        
        if is_arm_platform():
            self.logger.info("Raspberry Pi / ARM platform detected")
        
        # Initialize components
        self._init_components()
        
        # Runtime state
        self._running = Event()
        self._frame_count = 0
        self._total_detections = 0
        self._fps = 0.0
        self._processing_time = 0.0
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Configure logging based on config."""
        log_config = self.config.get('logging', {})
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
    
    def _init_components(self):
        """Initialize all detection and tracking components."""
        det_config = self.config.get('detection', {})
        track_config = self.config.get('tracking', {})
        dwell_config = self.config.get('dwell', {})
        zones_config = self.config.get('zones', [])
        api_config = self.config.get('api', {})
        
        # Person detector
        self.detector = PersonDetector(
            model_path=det_config.get('model', 'yolov8n.pt'),
            confidence=det_config.get('confidence', 0.5),
            device=det_config.get('device', 'auto'),
            use_ncnn=det_config.get('use_ncnn', False),
        )
        
        # Object tracker
        self.tracker = ObjectTracker(
            max_age=track_config.get('max_age', 30),
            min_hits=track_config.get('min_hits', 3),
            iou_threshold=track_config.get('iou_threshold', 0.3)
        )
        
        # Dwell tracker
        self.dwell_tracker = DwellTracker(
            min_dwell_time=dwell_config.get('min_dwell_time', 5.0)
        )
        
        # Zone manager
        self.zone_manager = ZoneManager(zones_config)
        
        # API server
        self.api_server = APIServer(
            host=api_config.get('host', '0.0.0.0'),
            port=api_config.get('port', 8000),
            cors_origins=api_config.get('cors_origins', ['*'])
        )
        self.api_server.setup(self.zone_manager, self.dwell_tracker)
        
        self.logger.info(f"Initialized with {len(zones_config)} monitoring zones")
    
    def _open_video_source(self):
        """Open the video source (USB webcam, RTSP, file, or Pi Camera)."""
        video_config = self.config.get('video', {})
        source = video_config.get('source', 0)
        width = video_config.get('width', 640)
        height = video_config.get('height', 480)
        fps = video_config.get('fps', 30)
        
        self.logger.info(f"Opening video source: {source}")
        
        # Raspberry Pi Camera Module
        if isinstance(source, str) and source.lower() in ("picamera", "picamera2", "pi"):
            self.logger.info("Using Raspberry Pi Camera Module via picamera2")
            return PiCameraSource(width=width, height=height, fps=fps)
        
        # OpenCV video source (webcam index, RTSP URL, or file)
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        # Set properties if specified
        if 'width' in video_config:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_config['width'])
        if 'height' in video_config:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_config['height'])
        if 'fps' in video_config:
            cap.set(cv2.CAP_PROP_FPS, video_config['fps'])
            
        return cap
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single video frame.
        
        Args:
            frame: BGR image
            
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        
        # Detect persons
        detections = self.detector.detect(frame)
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        frame_size = (width, height)
        
        # Check zone occupancy
        zone_occupancy = self.zone_manager.get_zone_occupancy(tracks, frame_size)
        
        # Update dwell tracking
        active_dwells = self.dwell_tracker.update(tracks, zone_occupancy)
        
        # Calculate metrics
        self._processing_time = (time.time() - start_time) * 1000
        self._total_detections += len(detections)
        self._frame_count += 1
        
        # Prepare detection data for API
        detection_data = []
        for track in tracks:
            track_zone = None
            for zone_name, track_ids in zone_occupancy.items():
                if track["track_id"] in track_ids:
                    track_zone = zone_name
                    break
            
            detection_data.append({
                "track_id": track["track_id"],
                "bbox": track["bbox"],
                "zone": track_zone
            })
        
        # Update API server
        self.api_server.update_detections(detection_data)
        self.api_server.update_stats(
            total_detections=self._total_detections,
            active_tracks=len(tracks),
            fps=self._fps,
            processing_time_ms=self._processing_time
        )
        
        return {
            "detections": detections,
            "tracks": tracks,
            "zone_occupancy": zone_occupancy,
            "active_dwells": active_dwells,
            "processing_time_ms": self._processing_time
        }
    
    def run_video_loop(self):
        """Run the main video processing loop."""
        cap = self._open_video_source()
        
        # Warm up detector
        video_config = self.config.get('video', {})
        warmup_h = video_config.get('height', 480)
        warmup_w = video_config.get('width', 640)
        
        self.logger.info("Warming up detector...")
        self.detector.warmup(input_shape=(warmup_h, warmup_w, 3))
        
        self.logger.info("Starting video processing loop")
        
        fps_start_time = time.time()
        fps_frame_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 30
        
        # Frame skip for low-power devices
        frame_skip = self.config.get('video', {}).get('frame_skip', 1)
        raw_frame_count = 0
        
        try:
            while self._running.is_set():
                ret, frame = cap.read()
                
                if not ret:
                    consecutive_failures += 1
                    
                    # For video files, optionally loop
                    source = video_config.get('source', 0)
                    
                    if isinstance(source, str) and Path(source).exists():
                        self.logger.info("Video ended, restarting...")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        consecutive_failures = 0
                        continue
                    
                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error(
                            f"Failed to read {max_consecutive_failures} consecutive frames. "
                            f"Video source may be disconnected. Stopping."
                        )
                        break
                    
                    # Exponential backoff: 0.1s, 0.2s, 0.4s, ... capped at 5s
                    backoff = min(0.1 * (2 ** (consecutive_failures - 1)), 5.0)
                    self.logger.warning(
                        f"Failed to read frame (attempt {consecutive_failures}/{max_consecutive_failures}), "
                        f"retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    continue
                
                # Reset failure counter on successful read
                consecutive_failures = 0
                
                # Frame skip — only process every Nth frame
                raw_frame_count += 1
                if frame_skip > 1 and (raw_frame_count % frame_skip) != 0:
                    continue
                
                # Process frame
                results = self.process_frame(frame)
                
                # Calculate FPS
                fps_frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    self._fps = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                # Log periodic stats
                if self._frame_count % 100 == 0:
                    self.logger.info(
                        f"Frame {self._frame_count}: "
                        f"FPS={self._fps:.1f}, "
                        f"Tracks={len(results['tracks'])}, "
                        f"Processing={results['processing_time_ms']:.1f}ms"
                    )
                    
        finally:
            cap.release()
            self.logger.info("Video processing loop ended")
    
    async def run_async(self):
        """Run the server with async API."""
        self._running.set()
        
        # Start video processing in background thread
        video_thread = Thread(target=self.run_video_loop, daemon=True)
        video_thread.start()
        
        # Run API server
        self.logger.info(f"Starting API server on {self.api_server.host}:{self.api_server.port}")
        await self.api_server.run()
    
    def run(self):
        """Run the server (blocking)."""
        def signal_handler(sig, frame):
            self.logger.info("Shutdown signal received")
            self._running.clear()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        asyncio.run(self.run_async())
    
    def stop(self):
        """Stop the server."""
        self._running.clear()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Computer Vision Server - Customer Detection & Dwell Tracking"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        help="Override video source (webcam index, RTSP URL, file path, or 'picamera')"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        help="Override API port"
    )
    parser.add_argument(
        "--raspberry", "--rpi",
        action="store_true",
        help="Use Raspberry Pi optimized config (configs/raspberry.yaml)"
    )
    
    args = parser.parse_args()
    
    # Use RPi config if --raspberry flag is set
    config_path = args.config
    if args.raspberry:
        rpi_config = Path("configs/raspberry.yaml")
        if rpi_config.exists():
            config_path = str(rpi_config)
        else:
            print(f"Warning: {rpi_config} not found, using {config_path}")
    
    # Create and run server
    server = VisionServer(config_path)
    
    # Apply command-line overrides
    if args.source is not None:
        try:
            server.config['video']['source'] = int(args.source)
        except ValueError:
            server.config['video']['source'] = args.source
    
    if args.port is not None:
        server.config['api']['port'] = args.port
        server.api_server.port = args.port
    
    server.run()


if __name__ == "__main__":
    main()
