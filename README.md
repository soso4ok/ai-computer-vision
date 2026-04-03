# AI Computer Vision Server

AI-powered customer detection and dwell-time tracking system built for **Raspberry Pi** and standard Linux servers. Uses YOLOv8 with NCNN optimization for real-time person detection on ARM hardware.

## Features

- **Real-time Person Detection** — YOLOv8 nano with NCNN export for fast ARM inference
- **Raspberry Pi Native** — Supports Pi Camera Module (via `picamera2`) and USB webcams
- **Dwell-Time Tracking** — Monitors how long customers spend in designated zones
- **Zone Configuration** — Define custom polygonal monitoring zones
- **REST API & WebSocket** — FastAPI-based analytics dashboard with real-time streaming
- **Frame Skip** — Configurable frame skipping for low-power devices
- **Docker Deployment** — Multi-arch images (ARM64 + AMD64)

## Architecture

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Video Source        │────▶│  Detection Engine │────▶│  Analytics API  │
│  USB / Pi Camera     │     │  YOLOv8 + NCNN   │     │  (FastAPI)      │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
                                    │
                                    ▼
                            ┌──────────────────┐
                            │  Dwell Tracker   │
                            │  (Zone Analysis) │
                            └──────────────────┘
```

## Hardware Requirements

| Board | RAM | Performance |
|-------|-----|-------------|
| Raspberry Pi 5 | 4GB+ | ~8-12 FPS (recommended) |
| Raspberry Pi 4 | 4GB+ | ~3-6 FPS |
| x86 Linux Server | 8GB+ | ~25-30 FPS (with GPU: 60+) |

**Camera options:**
- USB webcam (any UVC-compatible camera)
- Raspberry Pi Camera Module v2/v3 (via `picamera2`)
- RTSP network camera stream

## Quick Start

### Raspberry Pi Setup

```bash
# 1. Install system dependencies
sudo apt update && sudo apt install -y \
    python3-pip python3-venv \
    libgl1-mesa-glx libglib2.0-0 ffmpeg \
    python3-picamera2   # Only for Pi Camera Module

# 2. Clone and install
git clone https://github.com/soso4ok/ai-computer-vision.git
cd ai-computer-vision
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run with Raspberry Pi optimized config
python src/main.py --raspberry

# Or explicitly:
python src/main.py --config configs/raspberry.yaml
```

### Using Pi Camera Module

```yaml
# In configs/raspberry.yaml, set:
video:
  source: "picamera"    # Uses picamera2 library
  width: 640
  height: 480
  fps: 15
```

### Using USB Webcam

```yaml
# In configs/raspberry.yaml, set:
video:
  source: 0             # /dev/video0
  width: 640
  height: 480
  fps: 15
```

### Standard Linux Server

```bash
pip install -r requirements.txt
python src/main.py --config configs/default.yaml
```

### Using Docker

```bash
# Build (auto-detects ARM64 or AMD64)
docker compose build

# Run with USB webcam
docker compose up -d

# Run on Raspberry Pi with Pi Camera
# Uncomment /dev/vchiq in docker-compose.yml, then:
docker compose up -d
```

## Configuration

Two config presets are included:

| Config | Resolution | FPS | Frame Skip | NCNN | Use Case |
|--------|-----------|-----|-----------|------|----------|
| `configs/default.yaml` | 1280×720 | 30 | 1 (none) | No | x86 server / GPU |
| `configs/raspberry.yaml` | 640×480 | 15 | 2 | Yes | Raspberry Pi 4/5 |

### Key Settings

```yaml
video:
  source: 0              # 0=webcam, "picamera"=Pi Camera, "rtsp://..."
  frame_skip: 2           # Process every 2nd frame (saves CPU)

detection:
  model: "yolov8n.pt"     # Nano model (fastest)
  confidence: 0.4         # Lower = more detections, more false positives
  use_ncnn: true          # NCNN export for ARM (auto on Raspberry Pi)

zones:
  - name: "entrance"
    points: [[0,0.6], [0.3,0.6], [0.3,1.0], [0,1.0]]
```

## Project Structure

```
ai-computer-vision/
├── src/
│   ├── main.py              # Entry point (PiCamera + webcam support)
│   ├── detector.py          # YOLOv8 detection (NCNN ARM export)
│   ├── tracker.py           # Kalman filter object tracking
│   ├── dwell_tracker.py     # Dwell-time calculation
│   ├── zone_manager.py      # Zone configuration and monitoring
│   └── api.py               # REST API endpoints
├── configs/
│   ├── default.yaml         # Standard server config
│   ├── raspberry.yaml       # Raspberry Pi optimized config
│   └── prometheus.yml       # Metrics collection config
├── tests/                   # Unit + integration tests (50 tests)
├── scripts/
│   ├── deploy.sh            # Deployment automation
│   └── setup_server.sh      # Server provisioning
├── Dockerfile               # Multi-arch (ARM64 + AMD64)
├── docker-compose.yml
└── requirements.txt
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Current detection statistics |
| `/zones` | GET | List configured zones |
| `/zones/{name}` | GET | Zone details |
| `/zones/{name}/dwell` | GET | Dwell-time stats for a zone |
| `/zones/{name}/visitors` | GET | Current visitors in a zone |
| `/detections` | GET | Current tracked detections |
| `/detections/stream` | WS | Real-time detection stream |

## CLI Options

```bash
python src/main.py [OPTIONS]

Options:
  --config, -c PATH    Config file (default: configs/default.yaml)
  --source, -s SOURCE  Override video source (0, "picamera", rtsp://...)
  --port, -p PORT      Override API port
  --raspberry, --rpi   Use configs/raspberry.yaml automatically
```

## Troubleshooting

**Pi Camera not detected:**
```bash
# Check camera is enabled
sudo raspi-config  # Interface Options → Camera → Enable
# Verify camera works
libcamera-hello
```

**Low FPS on Raspberry Pi:**
- Increase `frame_skip` to 3 or 4
- Reduce resolution to 320×240
- Ensure NCNN is enabled (`use_ncnn: true`)
- Close other applications to free RAM

**Model download slow on first run:**
```bash
# Pre-download the model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**USB webcam not found:**
```bash
# Check available devices
ls -la /dev/video*
# If using Docker, uncomment the device mapping in docker-compose.yml
```

## System Requirements

- **Raspberry Pi**: Pi 4 (4GB+) or Pi 5, Raspberry Pi OS (Bookworm)
- **Linux Server**: Ubuntu 20.04+, Python 3.9+
- **Optional**: CUDA GPU for accelerated inference on x86
- **Optional**: Docker & Docker Compose

## License

MIT License
