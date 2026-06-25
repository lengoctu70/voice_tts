# VieNeu-TTS Code Standards & Conventions

## Python Version & Environment

- **Minimum Python**: 3.10 (3.11, 3.12, 3.13 tested and supported)
- **Dependency Management**: `uv` (preferred) or `pip`
- **Virtual Environments**: uv manages automatically; manual venv supported

**Installation Variants**:
- **Torch-free (Core)**: `uv sync` → ONNX Runtime + sea-g2p
- **GPU**: `uv sync --group gpu` → PyTorch + transformers + lmdeploy
- **Development**: `uv sync --group dev` → pytest + type checkers

## Module Structure & Organization

### Layered Architecture

```
UI/API (gradio_main.py, web_stream.py)
    ↓
Factory Dispatcher (factory.py)
    ↓
Engine Backends (v3turbo.py, standard.py, fast.py, remote.py, xpu.py)
    ↓
Base Interface (base.py)
    ↓
Sub-engines (_v3_turbo_engine/, v3_turbo_serve/)
    ↓
Utilities (vieneu_utils/, core_utils.py)
    ↓
External Codecs & Phonemizers (neucodec, sea-g2p, MOSS)
```

### File Naming Conventions

- **Python modules**: `snake_case.py` (e.g., `v3turbo.py`, `phonemize_text.py`)
- **Documentation**: `kebab-case.md` (e.g., `project-overview-pdr.md`, `code-standards.md`)
- **Config files**: lowercase (e.g., `config.yaml`, `pyproject.toml`)
- **Descriptive names**: Prefer clarity over brevity. Example: `inference_v3_turbo.py` is better than `inference.py`

### Class Naming

- **Engine classes**: `CamelCase` with "VieNeuTTS" or "V3Turbo" prefix
  - `V3TurboVieNeuTTS` — v3 Turbo implementation
  - `VieNeuTTS` — v1 Standard
  - `FastVieNeuTTS` — v2 LMDeploy path
  - `RemoteVieNeuTTS` — API client
- **Model classes**: `CamelCase` matching transformer architecture
  - `V3TurboTransformer`
  - `TransformerLayer`
  - `SelfAttention`

### Method Naming

- **Public API**: snake_case, action-oriented
  - `infer(text, voice=None)` — synthesis entry point
  - `save(audio, path)` — file output
  - `list_preset_voices()` — enumeration
- **Internal helpers**: Prefix with `_` (single underscore = private, convention)
  - `_load_ref_mono(path)` — reference audio loading
  - `_split_text_chunks(text, max_chars)` — text segmentation

## Dependency Management Strategy

### Core Dependencies (Always Installed)
```
sea-g2p>=0.7.6          # G2P phonemization (torch-free, Rust-based)
onnxruntime>=1.20.0     # ONNX inference (CPU-optimal)
numpy                   # Numerical arrays
soundfile               # Audio I/O
soxr                    # Resampling (faster than librosa)
tokenizers>=0.20        # Phoneme tokenization
huggingface_hub         # Model downloading
PyYAML                  # config.yaml parsing
gradio>=5.49.1          # Web UI framework
perth>=0.2.0            # Optional watermarking
```

### GPU Optional Dependency Group `[gpu]`
```
torch                   # PyTorch (GPU inference)
torchaudio              # Audio processing
transformers            # HuggingFace models
librosa>=0.11.0         # Audio analysis (slower fallback)
neucodec>=0.0.4         # NeuCodec for v1/v2
lmdeploy                # Fast inference server (non-macOS)
llama-cpp-python        # GGUF quantized models
triton                  # GPU compiler (Linux/Windows)
accelerate              # Multi-GPU support
```

### Development Group `[dev]`
```
pytest                  # Unit testing framework
pytest-asyncio          # Async test support
librosa, transformers, neucodec  # GPU backend testing
```

**Rationale**: Minimal install keeps package <50MB; GPU/dev groups added only when needed.

### ONNX vs PyTorch Dual-Engine Pattern

**Factory Logic** (factory.py):
```python
def Vieneu(mode="v3turbo", **kwargs):
    match mode:
        case "v3turbo":
            from .v3turbo import V3TurboVieNeuTTS
            return V3TurboVieNeuTTS(**kwargs)
        case "standard":
            from .standard import VieNeuTTS
            return VieNeuTTS(**kwargs)
        # ... other modes
```

**Device Detection** (v3turbo.py:44–50):
```python
if device in (None, "auto"):
    try:
        import torch
        dev_type = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev_type = "cpu"  # Torch-free path
```

**Inference Selection**:
- **CPU + torch-free**: ONNX Runtime (via `_v3_turbo_engine/onnx_runtime_lite.py`)
- **GPU available**: PyTorch (via `_v3_turbo_engine/modeling_v3_turbo.py`)
- **Remote**: HTTP API (via `remote.py`)

## Configuration & State Management

### Global Config (config.yaml)
```yaml
text_settings:
  max_chars_per_chunk: 256        # Max chars per synthesis call
  max_total_chars_streaming: 3000 # Max for full streaming passage

backbone_configs:
  "VieNeu-TTS-v3-Turbo (Thử nghiệm)":
    repo: pnnbao-ump/VieNeu-TTS-v3-Turbo
    supports_streaming: false
    description: "48kHz, voice cloning, emotion cues"
```

**Loading** (base.py:57):
```python
self.normalizer = Normalizer()  # sea-g2p with punc_norm=True
```

### Environment Variables
- `HF_TOKEN` — Hugging Face API token (for gated models)
- `CUDA_VISIBLE_DEVICES` — GPU selection (Docker/training)
- `DEVICE` — Force device (cpu/cuda/mps)
- `PYTHONPATH` — Include `src/` during testing

## Code Organization by Concern

### Voice Management
- **Preset voices**: JSON registry in `src/vieneu/assets/voices.json`
- **Voice spec**: [Voice Preset Specification v1.0](https://github.com/pnnbao97/VieNeu-TTS/issues)
- **Storage**: `_preset_voices` dict keyed by voice_id
- **API**: `list_preset_voices()`, `get_preset_voice(voice_id)`

### Text Processing Pipeline
1. **Punctuation normalization** (sea-g2p `Normalizer`)
2. **Phonemization** (`phonemize_text_with_emotions()`)
3. **Emotion tag extraction** (`[cười]`, `[thở dài]`)
4. **Chunking** (`split_text_into_chunks()`)

**File**: `src/vieneu_utils/phonemize_text.py` (148 LOC)

### Audio I/O & Codec Handling

**Loading** (base.py:_load_ref_mono):
- Prefers `librosa` if [gpu] installed
- Falls back to `soundfile` + `soxr` (core deps)
- Returns float32 mono at target sample rate

**Codec Detection** (base.py:_load_codec):
```python
if "onnx" in codec_repo.lower() or "vieneu-codec" in codec_repo:
    # Use NeuCodecOnnx (CPU-only)
else:
    # Use neucodec package (GPU, if available)
```

**Sample Rates**:
- v1/v2: 24 kHz
- v3 Turbo: 48 kHz

### Batched Inference (v3 Turbo Serving)

**File**: `src/vieneu/v3_turbo_serve/` (376 LOC)

**Pipeline**:
1. Group texts by speaker/emotion → batch
2. Backbone: `V3TurboTransformer.forward()` → acoustic codes
3. Acoustic codec: `MOSS-Audio-Tokenizer-Nano.decode()` → waveforms
4. Concatenate with optional overlap handling

**CUDA Graph Optimization** (cudagraph.py):
- First run: capture static computation graph
- Subsequent runs: replay without Python overhead
- ~20–30% latency improvement

## Testing Strategy

### Framework
- **Test runner**: pytest
- **Async tests**: pytest-asyncio (for streaming paths)
- **Coverage tool**: pytest-cov (via Makefile)

### Test Organization
```
tests/
  test_factory.py           # Mode dispatch
  test_engine_turbo.py      # v3 Turbo (ONNX + PyTorch)
  test_engine_standard.py   # v1/v2 GPU
  test_engine_fast.py       # LMDeploy fast path
  test_engine_remote.py     # API client
  test_utils.py             # Utilities (phonemization, audio)
  test_base_utils.py        # Base class methods
```

### Test Configuration (pyproject.toml)
```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

**Running Tests**:
```bash
uv run pytest tests/
uv run pytest tests/test_factory.py -v
pytest tests/ --cov=src/vieneu  # Coverage report
```

### Marker-Based Skipping (Optional)
```python
@pytest.mark.gpu
def test_cuda_inference():
    # GPU-only tests, skipped without GPU
    pass
```

## Phonemization & Text Normalization

### sea-g2p Integration
**File**: `src/vieneu_utils/phonemize_text.py` (148 LOC)

**Key Function**:
```python
def phonemize_text_with_emotions(text, lang="vi"):
    """
    Extract phonemes + emotion tags from text.
    
    Args:
        text: Input text with optional [emotion] tags
        lang: "vi" or "en"
    
    Returns:
        (phonemes, emotion_tags)
    
    Emotion tag patterns: [cười], [thở dài], [hắng giọng]
    """
```

**Fallback Logic**:
- Try sea-g2p phonemization
- If fails: split on whitespace (graceful degradation)

### Punctuation Normalization
- Enabled via `sea-g2p.Normalizer(punc_norm=True)` (always on)
- Removes/normalizes Vietnamese diacritics
- Handles contractions

## Build & Packaging

### Entry Points (pyproject.toml)
```toml
[project.scripts]
vieneu-web = "apps.gradio_main:main"
vieneu-stream = "apps.web_stream:main"
```

**Installation**:
```bash
pip install vieneu
vieneu-web  # Runs Gradio UI on http://localhost:7860
vieneu-stream  # Runs FastAPI server on http://localhost:8000
```

### Makefile Targets
| Target | Purpose |
|--------|---------|
| `make install` | `uv sync` (core) or `--group gpu` |
| `make web` | Run `vieneu-web` Gradio UI |
| `make test` | Run pytest suite |
| `make lint` | Ruff linter + type check |
| `make docker-build` | Build Docker image |
| `make docker-run` | Run Docker with GPU support |
| `make clean` | Remove build artifacts |

### Docker Build Strategy

**Multi-stage approach**:
1. Base: `python:3.12-slim` + CUDA 12.8
2. Install uv + `uv sync --group gpu`
3. Copy source code
4. Optional: mount model weights (`:ro`)
5. Expose port 23333 (LMDeploy)

**Usage**:
```bash
docker build -t vieneu:latest .
docker run --gpus all -p 23333:23333 vieneu:latest
```

## Performance & Optimization Patterns

### Text Chunking
**Rationale**: Transform long texts into manageable chunks for parallel processing.

**Default**: 256 characters per chunk (config.yaml)

**Implementation** (core_utils.py):
```python
def split_text_into_chunks(text, max_chars=256):
    """Split on sentence boundaries; fallback to char limit."""
    # Try sentence splitting first (via sea-g2p or spacy)
    # Fallback: split on space/newline
```

### Audio Resampling
**Preference order**:
1. `librosa` (if [gpu] installed) — slower, comprehensive
2. `soxr` (core dependency) — fast, high-quality
3. `scipy.signal` (fallback) — basic quality

### Model Caching
- **HF cache**: `~/.cache/huggingface/` (auto-managed by huggingface_hub)
- **Custom cache**: Optional `cache_dir` parameter in engine __init__
- **Watermark caching**: Optional via perth library (gracefully skipped if missing)

## Logging & Debugging

### Logger Configuration
```python
import logging
logger = logging.getLogger("Vieneu.V3Turbo")  # Module-level logger
logger.info(f"Loading model from {repo}...")
```

**Logger names**:
- `"Vieneu"` — Base class (base.py)
- `"Vieneu.V3Turbo"` — v3 Turbo (v3turbo.py)
- `"Vieneu.Standard"` — v1 Standard (standard.py)
- `"Vieneu.Remote"` — Remote API (remote.py)

### Error Handling Patterns

**Device fallback** (v3turbo.py:44–50):
```python
try:
    import torch
    dev_type = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    dev_type = "cpu"  # Graceful fallback if torch import fails
```

**Codec loading fallback** (base.py:_load_codec):
```python
try:
    # Try ONNX decoder first
    from .utils import NeuCodecOnnx
    self.codec = NeuCodecOnnx.from_pretrained(codec_repo)
except Exception as e:
    logger.warning(f"Failed ONNX: {e}. Trying neucodec package...")
    # Fall back to PyTorch NeuCodec
```

## Documentation Standards

### Docstring Format (Google Style)
```python
def infer(self, text, voice=None, ref_audio=None, emotion="natural"):
    """
    Synthesize speech from text.
    
    Args:
        text: Input Vietnamese/English text (max 256 chars recommended)
        voice: Preset voice ID or custom via ref_audio
        ref_audio: Path to 3–5s reference clip for voice cloning
        emotion: "natural" (default) or "storytelling"
    
    Returns:
        numpy.ndarray: Audio waveform (float32, mono, sample_rate Hz)
    
    Raises:
        ValueError: If text is empty or voice not found
        RuntimeError: If model loading fails
    
    Example:
        >>> tts = Vieneu()
        >>> audio = tts.infer("Xin chào")
        >>> tts.save(audio, "output.wav")
    """
```

### Inline Comments
- **Use**: Complex logic, non-obvious tricks, architectural decisions
- **Avoid**: Obvious code (e.g., `x = x + 1  # increment x`)
- **Vietnamese OK**: Comments in Vietnamese are acceptable

### Type Hints
```python
from typing import Optional, Union, List, Dict, Any
import numpy as np

def infer(
    self,
    text: str,
    voice: Optional[str] = None,
    ref_audio: Optional[Union[str, Path]] = None,
    emotion: str = "natural"
) -> np.ndarray:
    """..."""
```

## Security & Best Practices

### Model Integrity
- All models sourced from Hugging Face (centralized, signed releases)
- Optional watermarking via perth library (graceful skip if missing)
- Gated models require HF_TOKEN (handled by huggingface_hub)

### Audio Privacy
- Models run fully locally; no telemetry
- Reference audio never sent to external services (except remote mode)
- Remote mode: encodes locally, sends codes only

### Dependency Pinning
- `uv.lock` provides reproducible builds across all platforms
- Critical deps pinned by minor version (e.g., `torch==2.8.0`)
- Codecs & phonemizers pinned explicitly

## Contribution Guidelines

### Code Review Checklist
- [ ] Follows snake_case (Python) / kebab-case (docs) naming
- [ ] Type hints present for public methods
- [ ] Docstrings (Google style) for classes/functions
- [ ] No hardcoded paths; use `Path` from `pathlib`
- [ ] Imports organized: stdlib → third-party → local
- [ ] Tests pass: `pytest tests/ -v`
- [ ] Linting passes: `ruff check src/`
- [ ] No torch imports outside `[gpu]` optional group
- [ ] ONNX fallback respected for CPU path

### Backward Compatibility
- Never change public API signatures without deprecation warnings
- Voice preset IDs must remain stable
- Config schema changes documented in CHANGELOG
- Minimum Python version bump requires release notes

---

**Last Updated**: 2026-06-25  
**Compliance**: Python ≥3.10, uv package manager, pytest testing
