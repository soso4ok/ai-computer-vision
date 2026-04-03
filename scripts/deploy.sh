#!/bin/bash
#
# AI Computer Vision Server - Deployment Script
# Handles building, running, and managing the Docker deployment
#

set -e

# Configuration
IMAGE_NAME="ai-vision-server"
CONTAINER_NAME="ai-vision"
DEFAULT_PORT=8000
DEFAULT_CONFIG="configs/default.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running or you don't have permission."
        exit 1
    fi
    
    log_info "Docker is available"
}

# Build the Docker image
build() {
    log_info "Building Docker image: ${IMAGE_NAME}"
    
    local build_args=""
    
    # Check for GPU support
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected, building with CUDA support"
        build_args="--build-arg USE_CUDA=1"
    fi
    
    docker build ${build_args} -t "${IMAGE_NAME}:latest" .
    
    log_info "Build completed successfully"
}

# Run the container
run() {
    local port="${1:-$DEFAULT_PORT}"
    local config="${2:-$DEFAULT_CONFIG}"
    local detached="${3:-true}"
    
    # Check if container already exists
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_warn "Container ${CONTAINER_NAME} already exists"
        read -p "Do you want to remove it and create a new one? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker rm -f "${CONTAINER_NAME}"
        else
            exit 0
        fi
    fi
    
    log_info "Starting container: ${CONTAINER_NAME}"
    
    local run_args="-p ${port}:8000"
    run_args+=" -v $(pwd)/configs:/app/configs:ro"
    run_args+=" --name ${CONTAINER_NAME}"
    
    # Add GPU support if available
    if command -v nvidia-smi &> /dev/null; then
        run_args+=" --gpus all"
        log_info "GPU support enabled"
    fi
    
    # Add webcam access
    if [ -e /dev/video0 ]; then
        run_args+=" --device=/dev/video0:/dev/video0"
        log_info "Webcam access enabled"
    fi
    
    if [ "$detached" = "true" ]; then
        run_args+=" -d"
    fi
    
    docker run ${run_args} "${IMAGE_NAME}:latest" --config "${config}"
    
    if [ "$detached" = "true" ]; then
        log_info "Container started in detached mode"
        log_info "API available at http://localhost:${port}"
        log_info "Use '$0 logs' to view logs"
    fi
}

# Stop the container
stop() {
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_info "Stopping container: ${CONTAINER_NAME}"
        docker stop "${CONTAINER_NAME}"
        log_info "Container stopped"
    else
        log_warn "Container ${CONTAINER_NAME} is not running"
    fi
}

# Remove the container
remove() {
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_info "Removing container: ${CONTAINER_NAME}"
        docker rm -f "${CONTAINER_NAME}"
        log_info "Container removed"
    else
        log_warn "Container ${CONTAINER_NAME} does not exist"
    fi
}

# Show container logs
logs() {
    local follow="${1:-false}"
    
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        if [ "$follow" = "true" ] || [ "$follow" = "-f" ]; then
            docker logs -f "${CONTAINER_NAME}"
        else
            docker logs "${CONTAINER_NAME}"
        fi
    else
        log_error "Container ${CONTAINER_NAME} is not running"
        exit 1
    fi
}

# Show container status
status() {
    echo "=== Docker Images ==="
    docker images | grep -E "(REPOSITORY|${IMAGE_NAME})" || echo "No images found"
    echo
    echo "=== Running Containers ==="
    docker ps | grep -E "(CONTAINER|${CONTAINER_NAME})" || echo "No containers running"
    echo
    
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "=== Container Health ==="
        local port=$(docker port "${CONTAINER_NAME}" 8000 2>/dev/null | cut -d: -f2)
        if [ -n "$port" ]; then
            if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
                log_info "API is healthy at http://localhost:${port}"
            else
                log_warn "API is not responding"
            fi
        fi
    fi
}

# Clean up Docker resources
clean() {
    log_info "Cleaning up Docker resources"
    
    # Remove container
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        docker rm -f "${CONTAINER_NAME}"
        log_info "Removed container: ${CONTAINER_NAME}"
    fi
    
    # Remove image
    if docker images --format '{{.Repository}}' | grep -q "^${IMAGE_NAME}$"; then
        docker rmi "${IMAGE_NAME}:latest"
        log_info "Removed image: ${IMAGE_NAME}"
    fi
    
    # Remove dangling images
    docker image prune -f
    
    log_info "Cleanup completed"
}

# Run with docker-compose
compose_up() {
    log_info "Starting with docker-compose"
    docker-compose up -d
    log_info "Services started"
}

compose_down() {
    log_info "Stopping docker-compose services"
    docker-compose down
    log_info "Services stopped"
}

# Print usage
usage() {
    echo "AI Computer Vision Server - Deployment Script"
    echo
    echo "Usage: $0 <command> [options]"
    echo
    echo "Commands:"
    echo "  build              Build the Docker image"
    echo "  run [port] [cfg]   Run the container (default port: 8000)"
    echo "  stop               Stop the running container"
    echo "  remove             Remove the container"
    echo "  logs [-f]          Show container logs (-f for follow)"
    echo "  status             Show status of containers and images"
    echo "  clean              Remove all related Docker resources"
    echo "  compose-up         Start with docker-compose"
    echo "  compose-down       Stop docker-compose services"
    echo "  help               Show this help message"
    echo
    echo "Examples:"
    echo "  $0 build                    # Build the Docker image"
    echo "  $0 run                      # Run with defaults"
    echo "  $0 run 9000                 # Run on port 9000"
    echo "  $0 run 8000 configs/prod.yaml  # Run with custom config"
    echo "  $0 logs -f                  # Follow logs"
}

# Main entry point
main() {
    check_docker
    
    case "${1:-help}" in
        build)
            build
            ;;
        run)
            run "$2" "$3" "$4"
            ;;
        stop)
            stop
            ;;
        remove)
            remove
            ;;
        logs)
            logs "$2"
            ;;
        status)
            status
            ;;
        clean)
            clean
            ;;
        compose-up)
            compose_up
            ;;
        compose-down)
            compose_down
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            log_error "Unknown command: $1"
            usage
            exit 1
            ;;
    esac
}

main "$@"
