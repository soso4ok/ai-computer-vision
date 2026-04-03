#!/bin/bash
#
# AI Computer Vision Server - Server Provisioning Script
# Sets up a bare-metal Linux server for running the AI application
#

set -e

# Configuration
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"
INSTALL_CUDA="${INSTALL_CUDA:-false}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Please run as root or with sudo"
        exit 1
    fi
}

# Detect OS
detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        OS_VERSION=$VERSION_ID
    else
        log_error "Cannot detect OS. /etc/os-release not found."
        exit 1
    fi
    
    log_info "Detected OS: $OS $OS_VERSION"
}

# Update system packages
update_system() {
    log_step "Updating system packages"
    
    case $OS in
        ubuntu|debian)
            apt-get update
            apt-get upgrade -y
            ;;
        centos|rhel|fedora)
            dnf update -y || yum update -y
            ;;
        *)
            log_warn "Unknown OS, skipping system update"
            ;;
    esac
}

# Install base dependencies
install_base_deps() {
    log_step "Installing base dependencies"
    
    case $OS in
        ubuntu|debian)
            apt-get install -y \
                build-essential \
                curl \
                wget \
                git \
                ca-certificates \
                gnupg \
                lsb-release \
                software-properties-common \
                libgl1-mesa-glx \
                libglib2.0-0 \
                libsm6 \
                libxext6 \
                libxrender-dev \
                ffmpeg \
                v4l-utils
            ;;
        centos|rhel|fedora)
            dnf groupinstall -y "Development Tools" || yum groupinstall -y "Development Tools"
            dnf install -y \
                curl \
                wget \
                git \
                mesa-libGL \
                glib2 \
                libSM \
                libXext \
                libXrender \
                ffmpeg \
                v4l-utils || true
            ;;
    esac
}

# Install Docker
install_docker() {
    log_step "Installing Docker"
    
    if command -v docker &> /dev/null; then
        log_info "Docker is already installed"
        docker --version
        return
    fi
    
    case $OS in
        ubuntu|debian)
            # Remove old versions
            apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
            
            # Add Docker's official GPG key
            install -m 0755 -d /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/$OS/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            chmod a+r /etc/apt/keyrings/docker.gpg
            
            # Set up repository
            echo \
              "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/$OS \
              $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
              tee /etc/apt/sources.list.d/docker.list > /dev/null
            
            apt-get update
            apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            ;;
        centos|rhel|fedora)
            dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo || \
            yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            
            dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin || \
            yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            ;;
    esac
    
    # Start and enable Docker
    systemctl start docker
    systemctl enable docker
    
    log_info "Docker installed successfully"
    docker --version
}

# Install NVIDIA drivers and CUDA
install_cuda() {
    if [ "$INSTALL_CUDA" != "true" ]; then
        log_info "Skipping CUDA installation (set INSTALL_CUDA=true to install)"
        return
    fi
    
    log_step "Installing NVIDIA drivers and CUDA"
    
    # Check for NVIDIA GPU
    if ! lspci | grep -i nvidia > /dev/null; then
        log_warn "No NVIDIA GPU detected, skipping CUDA installation"
        return
    fi
    
    case $OS in
        ubuntu)
            # Add NVIDIA package repository
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(echo $OS_VERSION | tr -d '.')/x86_64/cuda-keyring_1.0-1_all.deb
            dpkg -i cuda-keyring_1.0-1_all.deb
            rm cuda-keyring_1.0-1_all.deb
            
            apt-get update
            apt-get install -y cuda-toolkit-${CUDA_VERSION//./-}
            apt-get install -y nvidia-driver-535
            ;;
        centos|rhel)
            dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${OS_VERSION}/x86_64/cuda-rhel${OS_VERSION}.repo
            dnf install -y cuda-toolkit-${CUDA_VERSION//./-}
            ;;
    esac
    
    # Set up environment variables
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/profile.d/cuda.sh
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
    
    log_info "CUDA installed. Please reboot for drivers to take effect."
}

# Install NVIDIA Container Toolkit
install_nvidia_docker() {
    if [ "$INSTALL_CUDA" != "true" ]; then
        return
    fi
    
    log_step "Installing NVIDIA Container Toolkit"
    
    case $OS in
        ubuntu|debian)
            curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
            
            apt-get update
            apt-get install -y nvidia-container-toolkit
            ;;
        centos|rhel|fedora)
            curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
                tee /etc/yum.repos.d/nvidia-container-toolkit.repo
            
            dnf install -y nvidia-container-toolkit || yum install -y nvidia-container-toolkit
            ;;
    esac
    
    # Configure Docker to use NVIDIA runtime
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    
    log_info "NVIDIA Container Toolkit installed"
}

# Install Python
install_python() {
    log_step "Installing Python $PYTHON_VERSION"
    
    case $OS in
        ubuntu|debian)
            add-apt-repository -y ppa:deadsnakes/ppa
            apt-get update
            apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev python3-pip
            ;;
        centos|rhel|fedora)
            dnf install -y python${PYTHON_VERSION//.} python${PYTHON_VERSION//.}-devel python3-pip || \
            yum install -y python${PYTHON_VERSION//.} python${PYTHON_VERSION//.}-devel python3-pip
            ;;
    esac
    
    # Update pip
    python3 -m pip install --upgrade pip
    
    log_info "Python installed"
    python3 --version
}

# Create application user
create_app_user() {
    log_step "Creating application user"
    
    local APP_USER="aivision"
    
    if id "$APP_USER" &>/dev/null; then
        log_info "User $APP_USER already exists"
    else
        useradd -m -s /bin/bash $APP_USER
        log_info "Created user: $APP_USER"
    fi
    
    # Add to docker group
    usermod -aG docker $APP_USER
    
    # Create application directory
    mkdir -p /opt/ai-vision
    chown -R $APP_USER:$APP_USER /opt/ai-vision
    
    log_info "Application directory: /opt/ai-vision"
}

# Configure firewall
configure_firewall() {
    log_step "Configuring firewall"
    
    # Check for ufw (Ubuntu)
    if command -v ufw &> /dev/null; then
        ufw allow 8000/tcp comment 'AI Vision API'
        ufw allow 9090/tcp comment 'AI Vision Metrics'
        log_info "UFW rules added"
        return
    fi
    
    # Check for firewalld (CentOS/RHEL)
    if command -v firewall-cmd &> /dev/null; then
        firewall-cmd --permanent --add-port=8000/tcp
        firewall-cmd --permanent --add-port=9090/tcp
        firewall-cmd --reload
        log_info "Firewalld rules added"
        return
    fi
    
    log_warn "No firewall detected, skipping configuration"
}

# Create systemd service
create_service() {
    log_step "Creating systemd service"
    
    cat > /etc/systemd/system/ai-vision.service << 'EOF'
[Unit]
Description=AI Computer Vision Server
After=docker.service
Requires=docker.service

[Service]
Type=simple
User=aivision
Group=docker
WorkingDirectory=/opt/ai-vision
ExecStartPre=/usr/bin/docker pull ai-vision-server:latest || true
ExecStart=/opt/ai-vision/scripts/deploy.sh run
ExecStop=/opt/ai-vision/scripts/deploy.sh stop
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    
    log_info "Systemd service created: ai-vision.service"
    log_info "Enable with: systemctl enable ai-vision"
    log_info "Start with: systemctl start ai-vision"
}

# Print summary
print_summary() {
    echo
    echo "========================================="
    echo "  Server Provisioning Complete"
    echo "========================================="
    echo
    echo "Installed components:"
    echo "  - Base development tools"
    echo "  - Docker & Docker Compose"
    [ "$INSTALL_CUDA" = "true" ] && echo "  - NVIDIA CUDA & Container Toolkit"
    echo "  - Python $PYTHON_VERSION"
    echo
    echo "Application user: aivision"
    echo "Application directory: /opt/ai-vision"
    echo
    echo "Next steps:"
    echo "  1. Copy application files to /opt/ai-vision"
    echo "  2. Run: cd /opt/ai-vision && ./scripts/deploy.sh build"
    echo "  3. Run: ./scripts/deploy.sh run"
    echo
    echo "Or enable as service:"
    echo "  systemctl enable --now ai-vision"
    echo
    [ "$INSTALL_CUDA" = "true" ] && echo "NOTE: Reboot required for NVIDIA drivers"
}

# Main function
main() {
    echo "========================================="
    echo "  AI Computer Vision Server Setup"
    echo "========================================="
    echo
    
    check_root
    detect_os
    update_system
    install_base_deps
    install_docker
    install_cuda
    install_nvidia_docker
    install_python
    create_app_user
    configure_firewall
    create_service
    print_summary
}

# Run if executed directly
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
