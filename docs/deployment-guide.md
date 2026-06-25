# VieNeu-TTS Deployment Guide

## Overview

VieNeu-TTS supports multiple deployment topologies, from local CLI to Docker GPU servers to edge devices. This guide covers setup, configuration, and best practices for each approach.

**Full Docker & LMDeploy guidance**: See [`docs/Deploy.md`](./Deploy.md) (comprehensive).

## Deployment Options at a Glance

| Option | Hardware | Setup Time | Latency | Use Case | Entry Point |
|--------|----------|-----------|---------|----------|-------------|
| **Local ONNX (CPU)** | Any CPU | ~5 min | 100–300ms | Development, edge devices | `vieneu-web` Gradio |
| **Local PyTorch (GPU)** | NVIDIA/Apple Silicon | ~10 min | <50ms | Fast local synthesis | `vieneu-web` Gradio |
| **Docker LMDeploy** | NVIDIA GPU | ~5 min | <30ms | Production server, remote clients | `docker run` |
| **FastAPI Stream** | CPU/GPU | ~5 min | <100ms/chunk | Real-time streaming | `vieneu-stream` |
| **Remote API Client** | Lightweight | ~1 min | Variable | Web apps, Colab, edge | Python SDK |

## 1. Local Installation & Web UI

### CPU Only (Torch-Free, ONNX Runtime)

**Best for**: Development, testing, CPU-only machines, macOS

```bash
# Install VieNeu-TTS (minimal, torch-free)
pip install vieneu

# Or with uv (faster, reproducible)
uv sync

# Run Web UI
vieneu-web
# Opens http://127.0.0.1:7860
```

**Inference Path**: CPU → ONNX Runtime (v3 Turbo) or PyTorch-free codec

**Requirements**:
- Python ≥3.10
- 2GB RAM
- ~500MB disk (model cache)

**Performance**:
- v3 Turbo ONNX: ~150–250ms per second of audio (modern CPU)
- Batch-friendly: Process up to 32 texts concurrently

### GPU Installation

#### NVIDIA CUDA

```bash
# Install with GPU support
uv sync --group gpu

# Or
pip install "vieneu[gpu]"

# Run Web UI
vieneu-web
```

**Requirements**:
- NVIDIA GPU with CUDA ≥12.8
- NVIDIA Toolkit installed
- PyTorch compatible with CUDA version

**Inference Path**: GPU → PyTorch v3 Turbo or v2 LMDeploy

**Performance**:
- v3 Turbo PyTorch: <50ms per second of audio
- Batched: Linear scaling up to batch size 32

#### Apple Silicon (MPS)

```bash
# Install with GPU support (Metal Performance Shaders)
uv sync --group gpu

vieneu-web
```

**Inference Path**: GPU → PyTorch with Apple MPS acceleration

**Performance**:
- v3 Turbo: ~80–150ms per second of audio
- Note: ONNX is actually faster on Apple; use CPU path if speed critical

### Intel Arc GPU

```bash
# Install with GPU support (limited; research phase)
uv sync --group gpu

# Use Intel XPU backend
Vieneu(mode="xpu")
```

**Note**: Intel GPU support via torch.xpu; limited testing. Report issues on GitHub.

## 2. Docker Deployment (Recommended for Production)

### Quick Start with GPU

```bash
# Requires: NVIDIA Docker runtime (nvidia-docker)
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

docker run --gpus all \
  -p 23333:23333 \
  -v huggingface_cache:/root/.cache/huggingface \
  pnnbao/vieneu-tts:latest \
  --tunnel
```

**What this does**:
- Runs v2 Standard (highest quality, GPU-optimized)
- Exposes API on port 23333
- Auto-creates public tunnel (via bore) for external access
- Mounts HF cache for model persistence

**Public Access**:
```bash
# Check container logs for public URL
docker logs <container_id> | grep "Public"
# Output: bore.pub:31631
# Use this in remote clients: http://bore.pub:31631/v1
```

### Custom Model & Configuration

```bash
# Run specific v3 Turbo model
docker run --gpus all \
  -p 23333:23333 \
  pnnbao/vieneu-tts:latest serve \
  --model pnnbao-ump/VieNeu-TTS-v3-Turbo

# Run locally fine-tuned model
docker run --gpus all \
  -p 23333:23333 \
  -v $(pwd)/models:/workspace/models:ro \
  pnnbao/vieneu-tts:latest serve \
  --model /workspace/models/merged_model
```

**Configuration via Environment**:
```bash
docker run --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_TOKEN=<your_hf_token> \
  -p 23333:23333 \
  pnnbao/vieneu-tts:latest
```

### Docker Compose Setup

See [`docker/docker-compose.yml`](../docker/docker-compose.yml) for multi-container orchestration.

**For detailed Docker instructions**: See [`docs/Deploy.md`](./Deploy.md)

## 3. FastAPI Streaming Server

### Local Streaming

```bash
# Install & run streaming server
uv sync
vieneu-stream

# Server runs on http://localhost:8000
# API endpoint: POST /synthesize
```

### Features
- Chunked audio streaming (real-time playback)
- Supports CPU & GPU backends
- Lightweight (minimal overhead)
- Compatible with v2 models (stable); v3 Turbo streaming experimental

### Usage from Client

```python
from vieneu import Vieneu

# Connect to streaming server
tts = Vieneu(mode="remote", api_base="http://localhost:8000/v1")

# Synthesis (streams audio in real-time)
text = "Chào bạn. Đây là một bài phát biểu dài..."
audio = tts.infer(text)
tts.save(audio, "output.wav")
```

## 4. Remote API Client (Lightweight)

### Connect to Running Server

**Server side** (e.g., Docker or `vieneu-stream`):
```bash
# Already running on bore.pub:31631 or your-domain.com
```

**Client side** (Web app, Colab, edge device):
```python
from vieneu import Vieneu

# Minimal install on client (no GPU, no torch)
# pip install vieneu

tts = Vieneu(
    mode="remote",
    api_base="http://bore.pub:31631/v1",  # or your server URL
    model_name="pnnbao-ump/VieNeu-TTS-v2"
)

# List available voices
for label, voice_id in tts.list_preset_voices():
    print(f"- {label} ({voice_id})")

# Synthesis (uses server GPU, client does no heavy lifting)
audio = tts.infer("Giọng được sinh ra trên máy chủ")
tts.save(audio, "output.wav")

# Voice cloning (encodes locally, sends codes to server)
audio = tts.infer(
    "Đây là giọng nhân bản",
    ref_audio="my_voice.wav",
    ref_text="Tác phẩm dự thi"
)
```

**Benefits**:
- Client: ~10MB install (torch-free)
- Server: 1 GPU handles 100s of clients
- Reference audio: Encoded locally (privacy-friendly)

### Colab Example

```python
# In Colab cell 1: Start server (background)
!pip install vieneu[gpu]
!nohup vieneu-stream > /dev/null 2>&1 &

# In Colab cell 2: Create tunnel (if needed)
import subprocess
subprocess.run(["pip", "install", "bore"])
# Use ngrok or similar for public tunnel

# In Colab cell 3: Client synthesis
!pip install vieneu
from vieneu import Vieneu
tts = Vieneu(mode="remote", api_base="http://localhost:8000/v1")
audio = tts.infer("Chào Colab!")
```

## 5. Edge & Embedded Deployment

### CPU-Only IoT Device

```bash
# Install (torch-free)
pip install vieneu

# Python script
from vieneu import Vieneu
import numpy as np

tts = Vieneu(mode="v3turbo")  # Auto-detects CPU → ONNX
audio = tts.infer("IoT synthesis")
tts.save(audio, "/tmp/output.wav")
```

**Footprint**:
- ONNX Runtime: ~50MB
- Models: ~300MB (v3 Turbo)
- Total: <500MB

**Latency**: ~200–500ms per second of text (depends on CPU)

### Raspberry Pi / ARM64 Board

```bash
# RPi 4 with 8GB+ RAM, 32GB+ storage

# Install dependencies
sudo apt update
sudo apt install python3.10 python3-pip libsndfile1

# Install VieNeu (may take 5–10 min)
pip install vieneu

# Test inference
python3 -c "from vieneu import Vieneu; tts = Vieneu(); audio = tts.infer('Test')"
```

**Performance**: ~500–1000ms per second of text (RPi 4)

**Limitation**: No GPU acceleration; use ONNX path only

## Configuration Management

### Environment Variables

| Variable | Default | Purpose | Example |
|----------|---------|---------|---------|
| `HF_TOKEN` | None | Hugging Face API (gated models) | `hf_***` |
| `CUDA_VISIBLE_DEVICES` | All | GPU selection (Docker) | `0,1` |
| `DEVICE` | auto | Force device (cpu/cuda/mps) | `cpu` |
| `PYTHONPATH` | — | Add src/ for local testing | `src/` |

**Usage**:
```bash
# Docker
docker run -e HF_TOKEN=$HF_TOKEN \
  -e CUDA_VISIBLE_DEVICES=0 \
  pnnbao/vieneu-tts:latest

# Python
import os
os.environ["DEVICE"] = "cuda"
from vieneu import Vieneu
tts = Vieneu(device="cuda")
```

### Config File (config.yaml)

**Path**: Project root `config.yaml`

**Sections**:
- `text_settings` — Chunk size, streaming limits
- `backbone_configs` — Available models
- `codec_configs` — Audio codecs

**Example customization**:
```yaml
text_settings:
  max_chars_per_chunk: 512  # Increase for longer chunks
  max_total_chars_streaming: 5000

backbone_configs:
  "VieNeu-TTS-v3-Turbo (Custom)":
    repo: your-org/your-custom-v3-turbo
    supports_streaming: true
```

## Performance Tuning

### Batching for Throughput

```python
from vieneu import Vieneu

tts = Vieneu(mode="v3turbo")

# Batch 8 texts for 5–8x speedup
texts = ["Text 1", "Text 2", ..., "Text 8"]
for text in texts:
    audio = tts.infer(text)
    # Processes in parallel on GPU
```

### Chunk Size Optimization

```python
# config.yaml
text_settings:
  max_chars_per_chunk: 256  # Default
  # Tune based on your hardware:
  # - CPU: 128–256 (keep latency low)
  # - GPU: 256–512 (maximize throughput)
```

### GPU Memory Management

```bash
# Monitor GPU usage
nvidia-smi -l 1  # Update every 1 sec

# Limit GPU memory (Docker)
docker run --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  pnnbao/vieneu-tts:latest
```

## Monitoring & Logging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("Vieneu")
logger.setLevel(logging.DEBUG)

from vieneu import Vieneu
tts = Vieneu(mode="v3turbo")
audio = tts.infer("Debug mode")  # Verbose output
```

### Server Health Checks

```bash
# Check Docker container
docker logs <container_id> | tail -20

# Check API endpoint
curl http://localhost:23333/v1/models  # If LMDeploy running

# Streaming server status
curl http://localhost:8000/status
```

### Performance Profiling

```python
from time import time
from vieneu import Vieneu

tts = Vieneu(mode="v3turbo")

# Time inference
start = time()
audio = tts.infer("Test text")
elapsed = time() - start
print(f"Latency: {elapsed*1000:.1f}ms")

# Throughput (texts/sec)
num_texts = 100
start = time()
for i in range(num_texts):
    tts.infer(f"Text {i}")
throughput = num_texts / (time() - start)
print(f"Throughput: {throughput:.1f} texts/sec")
```

## Troubleshooting

### Common Issues

#### PyTorch Import Error
```
ImportError: No module named 'torch'
```
**Solution**: Install GPU extras
```bash
uv sync --group gpu
# or
pip install "vieneu[gpu]"
```

#### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU path
```python
tts = Vieneu(device="cpu")  # Force CPU
# or in Docker:
# docker run -e CUDA_VISIBLE_DEVICES=0 pnnbao/vieneu-tts:latest
```

#### Model Download Timeout
```
huggingface_hub.utils._errors.RepositoryNotFoundError
```
**Solution**: Set HF token & check internet
```bash
export HF_TOKEN=<your_hf_token>
# or Docker:
docker run -e HF_TOKEN=$HF_TOKEN pnnbao/vieneu-tts:latest
```

#### Slow Inference
```
Latency >500ms per second (unexpectedly slow)
```
**Solution**: Check hardware & backend
```bash
# Verify GPU is being used
nvidia-smi  # Should show vieneu process

# Force GPU
tts = Vieneu(device="cuda")

# Check ONNX providers (CPU path)
from vieneu._v3_turbo_engine.onnx_runtime_lite import ONNXRuntimeLite
runtime = ONNXRuntimeLite()
print(runtime.get_providers())  # Should show optimized providers
```

## Best Practices

1. **Use Docker for production** — Reproducible, isolated, easy scaling
2. **Pin dependencies** — Use `uv.lock` for reproducibility
3. **Monitor GPU memory** — Set `CUDA_VISIBLE_DEVICES` per container
4. **Cache models** — Mount HF cache volume to persist downloads
5. **Use remote mode for scale** — Separate GPU server from lightweight clients
6. **Enable streaming** — For real-time UX, use `vieneu-stream`
7. **Test locally first** — Develop with `vieneu-web` before Docker deployment
8. **Keep torch-free path** — Default install stays lightweight (no torch)

## Next Steps

- **Full Docker guide**: [`docs/Deploy.md`](./Deploy.md)
- **Custom model fine-tuning**: [`docs/CUSTOM_MODEL_USAGE.md`](./CUSTOM_MODEL_USAGE.md)
- **Makefile automation**: `make web`, `make docker-build`, `make docker-run`
- **Examples**: `examples/main.py`, `examples/main_remote.py`

---

**Last Updated**: 2026-06-25  
**Deployment Version**: 3.0.5
