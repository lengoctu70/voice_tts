#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="docker/docker-compose.yml"
PROFILE="gpu"
SERVICE="gpu"

info()  { printf '\033[1;34m[INFO]\033[0m  %s\n' "$*"; }
warn()  { printf '\033[1;33m[WARN]\033[0m  %s\n' "$*"; }
error() { printf '\033[1;31m[ERROR]\033[0m %s\n' "$*"; exit 1; }

# --- Prerequisites ---
info "Checking prerequisites..."

command -v docker >/dev/null 2>&1 || error "Docker not installed. Install: https://docs.docker.com/engine/install/"
docker compose version >/dev/null 2>&1 || error "Docker Compose V2 not found. Install: https://docs.docker.com/compose/install/"
nvidia-smi >/dev/null 2>&1 || error "NVIDIA driver not detected. Install driver first."
docker info 2>/dev/null | grep -qi nvidia || warn "NVIDIA Container Toolkit may not be installed. Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"

info "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# --- Environment ---
if [ ! -f .env ]; then
    info "Creating .env from .env.example..."
    cp .env.example .env
    warn "Review .env and set HF_TOKEN if needed for private models."
fi

# --- Build ---
info "Building Docker image (this may take 10-20 min on first run)..."
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" build

# --- Stop existing ---
if docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" ps --status running 2>/dev/null | grep -q "$SERVICE"; then
    info "Stopping existing container..."
    docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" down
fi

# --- Start ---
info "Starting VieNeu-TTS with GPU support..."
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d

# --- Output ---
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || curl -s ifconfig.me 2>/dev/null || echo "localhost")
PORT=$(grep -E '^PORT=' .env 2>/dev/null | cut -d= -f2 || echo "7860")
PORT="${PORT:-7860}"

echo ""
info "========================================="
info "VieNeu-TTS is running!"
info "Local:   http://localhost:${PORT}"
info "Network: http://${HOST_IP}:${PORT}"
info "========================================="
echo ""
info "View logs:    docker compose -f $COMPOSE_FILE --profile $PROFILE logs -f"
info "Stop:         docker compose -f $COMPOSE_FILE --profile $PROFILE down"
info "Rebuild:      git pull && bash deploy-vps.sh"
