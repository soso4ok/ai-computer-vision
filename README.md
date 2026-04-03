# Bare-Metal AI Computer Vision Server

AI-powered customer detection and dwell-time tracking application designed for deployment on physical Linux servers.

## Features

- **Real-time Person Detection**: Uses YOLOv8 for accurate customer detection
- **Dwell-Time Tracking**: Monitors how long customers spend in designated zones
- **Zone Configuration**: Define custom monitoring zones via configuration
- **Analytics Dashboard**: REST API for accessing detection metrics
- **Docker Deployment**: Containerized for easy deployment and scaling

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Video Source   │────▶│  Detection Engine │────▶│  Analytics API  │
│  (Camera/RTSP)  │     │  (YOLOv8 + Track) │     │  (FastAPI)      │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │  Dwell Tracker   │
                        │  (Zone Analysis) │
                        └──────────────────┘
```

## Quick Start

### Using Docker (Recommended)

```bash
# Build and run
./scripts/deploy.sh build
./scripts/deploy.sh run

# Or use docker-compose
docker-compose up -d
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python src/main.py --config configs/default.yaml
```

## Configuration

Edit `configs/default.yaml` to configure:
- Video source (webcam, RTSP stream, video file)
- Detection confidence threshold
- Monitoring zones
- API settings

## Project Structure

```
ai-computer-vision/
├── src/
│   ├── main.py              # Application entry point
│   ├── detector.py          # YOLOv8 person detection
│   ├── tracker.py           # Object tracking and ID assignment
│   ├── dwell_tracker.py     # Dwell-time calculation
│   ├── zone_manager.py      # Zone configuration and monitoring
│   └── api.py               # REST API endpoints
├── scripts/
│   ├── deploy.sh            # Deployment automation
│   ├── setup_server.sh      # Server provisioning script
│   └── run_tests.py         # Test automation
├── configs/
│   └── default.yaml         # Default configuration
├── tests/
│   ├── test_detector.py     # Detection tests
│   └── test_tracker.py      # Tracking tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | Current detection statistics |
| `/zones` | GET | List configured zones |
| `/zones/{id}/dwell` | GET | Dwell-time data for zone |
| `/detections/stream` | WS | Real-time detection stream |

## System Requirements

- Linux (Ubuntu 20.04+ recommended)
- Python 3.9+
- CUDA-capable GPU (optional, for acceleration)
- Docker & Docker Compose

## License

MIT License
