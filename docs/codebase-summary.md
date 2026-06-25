# VieNeu-TTS Codebase Summary

## Repository Layout

| Directory | Purpose | Key Files |
|-----------|---------|-----------|
| `src/vieneu/` | Core TTS engine implementations | factory.py, base.py, v3turbo.py, standard.py, fast.py, turbo.py, remote.py |
| `src/vieneu/_v3_turbo_engine/` | v3 Turbo transformer & ONNX runtime | modeling_v3_turbo.py, inference_v3_turbo.py, onnx_runtime_lite.py |
| `src/vieneu/v3_turbo_serve/` | Batched serving pipeline for v3 Turbo | batched_backbone.py, batched_acoustic.py, cudagraph.py, engine.py |
| `src/vieneu_utils/` | Shared utilities | phonemize_text.py, core_utils.py, url_extract.py, srt_parser.py, srt_audio_ops.py, srt_to_audio.py |
| `apps/` | Web UI & streaming entry points | gradio_main.py (vieneu-web), web_stream.py (vieneu-stream), gradio_xpu.py, srt_ui_handler.py |
| `tests/` | Unit & integration tests | test_factory.py, test_engine_*.py, test_utils.py |
| `finetune/` | Fine-tuning & LoRA workflows | train.py, merge_lora.py, create_voices_json.py |
| `examples/` | SDK usage examples | main.py, main_v3turbo.py, main_remote.py |
| `docker/` | Docker build configs | Dockerfile, docker-compose.yml |
| `docs/` | Documentation | README.md, Deploy.md, CUSTOM_MODEL_USAGE.md |

**Configuration & Build**:
- `config.yaml` — Backbone & codec selection, text streaming limits
- `pyproject.toml` — Dependency groups (core, [gpu], [dev]), uv package manager config
- `Makefile` — Development tooling (lint, test, web, clean)
- `.python-version` — Python 3.10+ requirement
- `uv.lock` — Reproducible dependency lock file

## Core Modules: `src/vieneu/`

### Factory & Dispatcher (factory.py, 42 LOC)
**Purpose**: Single entry point for all backends via mode dispatch.

```python
Vieneu(mode="v3turbo", **kwargs)  # Returns appropriate engine instance
```

**Modes**:
- `"v3turbo"` (default) → `V3TurboVieNeuTTS` (CPU/GPU)
- `"standard"` → `VieNeuTTS` (GPU PyTorch)
- `"fast"` / `"gpu"` → `FastVieNeuTTS` (LMDeploy v2)
- `"turbo"` / `"turbo_gpu"` → `TurboVieNeuTTS` (GGUF/PyTorch)
- `"remote"` / `"api"` → `RemoteVieNeuTTS` (API client)
- `"xpu"` → `XPUVieNeuTTS` (Intel GPU)

### Abstract Base (base.py, 467 LOC)
**Purpose**: Common interface & shared functionality for all backends.

**Key Methods**:
- `infer(text, voice=None, ref_audio=None, ref_text=None, emotion="natural")` → Audio array
- `save(audio, path)` → Write WAV
- `list_preset_voices()` → [(label, voice_id), ...]
- `get_preset_voice(voice_id)` → Voice metadata dict

**Key Attributes**:
- `sample_rate` — 24 kHz (v1/v2) or 48 kHz (v3 Turbo)
- `_preset_voices` — Built-in voice registry
- `_ref_phoneme_cache` — Reference text phoneme cache
- `normalizer` — sea-g2p Normalizer for text preprocessing
- `watermarker` — Optional audio watermarking (perth library)

**Codec Loading**:
- Detects ONNX vs PyTorch codecs automatically
- v3 Turbo uses MOSS-Audio-Tokenizer-Nano codec (no separate codec_repo)
- Fallback for reference loading: librosa (if [gpu]) → soundfile + soxr (core)

### VieNeu-TTS v3 Turbo (v3turbo.py, 245 LOC)
**Purpose**: Default high-speed 48 kHz engine; CPU via ONNX Runtime, GPU via PyTorch.

**Initialization**:
```python
V3TurboVieNeuTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS-v3-Turbo",
    moss_tokenizer="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
    device="auto",  # CPU or cuda
    backend="auto",  # ONNX (CPU) or PyTorch (GPU)
)
```

**Features**:
- 48 kHz audio (via MOSS codec)
- Built-in speaker tokens (no reference needed)
- Voice cloning from 3–5s reference audio
- Emotion cues (experimental): `[cười]`, `[thở dài]`, `[hắng giọng]`
- Batched generation (batch size ≤32)
- Multi-speaker conversation mode

**Inference Path**:
1. Device detection (CPU → ONNX, GPU → PyTorch auto-detected)
2. Model loading from Hugging Face
3. Text chunking (max 256 chars per chunk)
4. Phonemization via sea-g2p + emotion tag parsing
5. Batched backbone inference
6. Acoustic codec decoding
7. Audio concatenation

### VieNeu-TTS v1 Standard (standard.py, 432 LOC)
**Purpose**: Original, stable backbone for maximum GPU quality.

**Features**:
- 24 kHz audio (NeuCodec)
- Vietnamese-only (no bilingual)
- Voice cloning support
- PyTorch inference on GPU

### VieNeu-TTS v2 Fast (fast.py, 256 LOC)
**Purpose**: Lightweight LMDeploy-based serving for fast GPU inference.

**Features**:
- LMDeploy backend (quantized, optimized)
- 24 kHz bilingual audio
- Podcast/conversation mode
- Streaming support

### VieNeu-TTS v2 Turbo (turbo.py, 418 LOC)
**Purpose**: Lightweight GGUF-based inference (GGML quantized models).

**Features**:
- GGUF format for CPU efficiency
- Bilingual code-switching
- Voice cloning
- Podcast mode

### Remote API Client (remote.py, 295 LOC)
**Purpose**: Lightweight client for remote LMDeploy server (no GPU on client).

**Usage**:
```python
tts = Vieneu(mode="remote", api_base="http://server:23333/v1")
audio = tts.infer("Text", voice="Bình An")
```

**Server Communication**:
- HTTP POST to `/v1/synthesize`
- Encodes ref audio locally, sends codes to server

### Shared Utilities (utils.py, 174 LOC)
**Purpose**: NeuCodec ONNX wrapper and helper functions.

- `NeuCodecOnnx` — ONNX codec decoder (CPU-only)
- `split_text_into_chunks()` — Text segmentation
- `join_audio_chunks()` — Audio concatenation

### Intel XPU Backend (core_xpu.py, 208 LOC)
**Purpose**: Intel GPU support via torch.xpu (Windows/Linux only).

**Initialization**:
```python
tts = Vieneu(mode="xpu")  # Auto-detects Intel GPU
```

**Features**:
- Bilingual support (v2 features)
- Podcast mode
- Streaming capability

## V3 Turbo Engine: `src/vieneu/_v3_turbo_engine/` (1095 LOC)

### Configuration (configuration_v3_turbo.py, ~44 LOC)
- Transformer config: 12 layers, 768 hidden, 3072 FFN, 8 heads
- Embedding dims, dropout rates, layer normalization params

### Modeling (modeling_v3_turbo.py, ~248 LOC)
**Transformer Architecture**:
- `V3TurboTransformer` — Main backbone with speaker token embedding (reserved IDs 13–42 for default voices)
- Self-attention layers with causal masking (for AR generation)
- Speaker embedding projection; emotion tag integration
- No explicit cross-attention (embeddings concatenated)

**Key Classes**:
- `SelfAttention` — Multi-head self-attention
- `FeedForward` — Position-wise MLP
- `TransformerLayer` — Single encoder/decoder block with residuals

### Inference (inference_v3_turbo.py, ~396 LOC)
**Purpose**: Orchestrate v3 Turbo inference (PyTorch path).

**Key Methods**:
- `forward(input_ids, speaker_ids, emotion_ids)` → Acoustic codes
- `generate()` — Auto-regressive generation with top-k/nucleus sampling
- `infer_batch()` — Batched inference for efficiency

**Features**:
- Speaker token embedding for built-in voices (ID 13–42 reserved)
- Emotion tag embedding (experimental): `[cười]`, `[thở dài]`, `[hắng giọng]`
- KV cache for efficient inference
- Temperature & top-k sampling

### ONNX Runtime (onnx_runtime_lite.py, ~306 LOC)
**Purpose**: Lightweight ONNX Runtime wrapper for CPU inference (torch-free).

**Classes**:
- `ONNXRuntimeLite` — Session manager with optimized provider selection
- Provider fallback: TensorRT → CUDA → CPU

**Key Methods**:
- `load_model(model_path)` → Load .onnx files
- `run(input_dict)` → Execute inference
- `get_providers()` → List available acceleration backends

### Prompt Engineering (prompt_v3_turbo.py, ~22 LOC)
**Purpose**: Speaker & emotion prompt templates.

- Default speaker ID mapping (Bình An, Ngọc Linh, Xuân Vĩnh, etc.)
- Emotion tag definitions (reserved codes)

### Hub Loading (hub_load_v3_turbo.py, ~49 LOC)
**Purpose**: Download & cache models from Hugging Face.

- Detects model type (ONNX vs PyTorch)
- Auto-selects ONNX on CPU if torch unavailable
- Respects `HF_TOKEN` for gated models

## V3 Turbo Serving: `src/vieneu/v3_turbo_serve/` (376 LOC)

**Purpose**: Optimized batched serving pipeline with CUDA graph support.

### Batched Backbone (batched_backbone.py, 64 LOC)
- Groups multiple inference calls into single batch
- Handles variable sequence lengths with padding
- Returns acoustic token codes

### Batched Acoustic Codec (batched_acoustic.py, 115 LOC)
- Decodes acoustic codes → waveforms in parallel
- MOSS codec decoder ONNX
- Supports GPU decoding for speed

### CUDA Graph Engine (cudagraph.py, 60 LOC)
**Purpose**: PyTorch CUDA graph optimization (GPU only).

- Captures inference graph on first run
- Replays for zero-copy subsequent runs
- ~20–30% speedup for repeated batch inference

### Engine Orchestrator (engine.py, 137 LOC)
**Purpose**: Coordinate batched inference, manage KV cache, handle streaming.

**Key Methods**:
- `infer_batch(texts, speakers, emotions)` → Batched audio generation
- `stream_infer(text, speaker)` → Streaming inference (experimental)
- `reset_cache()` — Clear KV cache between batches

## Utilities: `src/vieneu_utils/` (410 LOC)

### Phonemization (phonemize_text.py, 148 LOC)
**Purpose**: Text preprocessing + phonemization via sea-g2p.

**Key Functions**:
- `phonemize_text_with_emotions(text, lang="vi")` — Extract phonemes & emotion tags
- Emotion tag patterns: `[cười]`, `[thở dài]`, `[hắng giọng]`
- Fallback: split on whitespace if sea-g2p fails

### Core Utils (core_utils.py, 197 LOC)
**Key Functions**:
- `split_text_into_chunks(text, max_chars=256)` — Chunk for sequential processing
- `join_audio_chunks(chunks, overlap=0)` → Concatenate audio arrays
- `resample_audio(audio, sr_old, sr_new)` → soxr-based resampling
- `normalize_audio(audio)` → Loudness normalization

### URL Extraction (url_extract.py, 65 LOC)
- `extract_urls(text)` → Find HTTP(S) URLs
- Used by podcast mode to skip URL synthesis

### SRT Parser (srt_parser.py)
- `parse_srt(raw)` → `list[SrtBlock]`; validates ordering, overlap, duration, timestamp format, missing/blank text.
- `SrtValidationError(code, message, block=...)` — `code ∈ {empty, malformed_timestamp, missing_text, blank_text, bad_order, overlap, bad_duration}`.

### SRT Audio Ops (srt_audio_ops.py)
- `is_silent`, `trim_edge_silence`, `fit_to_window` (pad / time-stretch ≤1.25× / hard-cut), `assemble_timeline`.
- Uses `librosa.effects.time_stretch` when available; falls back to an in-module SOLA implementation.

### SRT-to-Audio Orchestrator (srt_to_audio.py)
- `synthesize_srt(text, synthesize_chunk, sr, ...)` generator: validates SRT, calls the supplied TTS callable per block (retries up to 2× on silent output), trims/fits, assembles the timeline.
- `ChunkLog` dataclass + `format_chunk_log(...)` for concise per-block diagnostics in the UI.

## Applications: `apps/` (3584 LOC)

### Web UI (gradio_main.py, 2204 LOC)
**Entry Point**: `vieneu-web` (via pyproject.toml script)

**Features**:
- Real-time synthesis via Gradio UI
- Backbone selection (config.yaml)
- Voice preset dropdown
- Voice cloning upload
- Emotion tag input field
- Audio playback & download
- Streaming mode toggle
- Conversation/podcast mode
- **SRT → Audio tab**: paste an SRT block list, generate one timeline-aligned WAV (delegates to `apps/srt_ui_handler.py` → `vieneu_utils.srt_to_audio.synthesize_srt`).

**UI Sections**:
1. Text input (max 3000 chars for streaming, 256 per chunk)
2. Voice selector (dropdown for presets or custom cloning)
3. Advanced options (emotion, speaker, batch size)
4. Output audio player

### Streaming Server (web_stream.py, 221 LOC)
**Entry Point**: `vieneu-stream`

**Features**:
- FastAPI server on port 8000
- HTTP streaming endpoint
- Bore tunnel support for public access
- Lightweight (no model loading on server startup)

**Endpoints**:
- `POST /synthesize` — Streaming synthesis
- `GET /status` — Server status

### Intel XPU UI (gradio_xpu.py, 847 LOC)
**Entry Point**: `gradio_xpu` (for Intel GPU users)

- Similar to gradio_main.py but uses XPU backend
- Windows batch file: `setup_xpu_uv.bat`, `run_xpu.bat`

### UI Constants (ui_constants.py, 157 LOC)
- Emotion tag list
- Speaker presets
- Model descriptions (from config.yaml)
- CSS styling

### UI Utilities (ui_utils.py, 154 LOC)
- Audio processing helpers
- Voice cloning input validation
- Streaming state management

## Testing: `tests/` (8 test files)

**Coverage**:
- `test_factory.py` — Mode dispatch & engine instantiation
- `test_engine_turbo.py` — v3 Turbo inference paths (ONNX + PyTorch)
- `test_engine_standard.py` — v1/v2 GPU backends
- `test_engine_fast.py` — LMDeploy fast path
- `test_engine_remote.py` — Remote API client
- `test_utils.py` — Utility functions (phonemization, audio)
- `test_base_utils.py` — Base class helpers

**Test Framework**: pytest + pytest-asyncio for async tests

## Fine-tuning: `finetune/` (7 scripts)

### Training (train.py)
- LoRA adapter training on custom data
- Supports v1/v2/v3 backbones
- Dataset loading & preprocessing
- Checkpoint saving

### LoRA Merge (merge_lora.py)
- Merge trained LoRA adapters into base model
- Output: standalone model weights
- Used before Docker deployment

### Voices JSON (create_voices_json.py)
- Generate voices.json from reference clips
- Speaker embedding extraction
- Voice metadata (name, description, gender, accent)

### Data Scripts
- `encode_data.py` — Preprocess raw audio
- `filter_data.py` — Data quality filtering
- `get_hf_sample.py` — Download Hugging Face datasets

## Configuration & Build

### config.yaml
```yaml
backbone_configs:
  "VieNeu-TTS-v3-Turbo (Thử nghiệm)":
    repo: pnnbao-ump/VieNeu-TTS-v3-Turbo
    description: "48kHz, CPU (ONNX) or GPU (PyTorch), voice cloning, emotion cues"
  "VieNeu-TTS-v2 (GPU)":
    repo: pnnbao-ump/VieNeu-TTS-v2
    description: "24kHz, bilingual, podcast mode, GPU only"
  "VieNeu-TTS (GPU)":
    repo: pnnbao-ump/VieNeu-TTS
    description: "24kHz, Vietnamese only, GPU, stable"

codec_configs:
  "NeuCodec (Distill)": neuphonic/distill-neucodec
  "VieNeu-Codec": pnnbao-ump/VieNeu-Codec
```

### pyproject.toml (5982 bytes)
**Key Sections**:
- `[project]` — Package metadata (v3.0.5, Python ≥3.10)
- `[project.dependencies]` — Core torch-free deps (sea-g2p, onnxruntime, gradio, etc.)
- `[project.optional-dependencies]` — `[gpu]` group (torch, transformers, etc.)
- `[dependency-groups]` — `[dev]` (pytest), `[gpu]` (extended for dev)
- `[project.scripts]` — Entry points (vieneu-web, vieneu-stream)
- `[tool.uv]` — Dependency groups, Python index config, required environments
- `[tool.pytest.ini_options]` — Test path configuration

**Dependency Strategy**:
- Minimal install: `pip install vieneu` (torch-free, ONNX only)
- GPU install: `uv sync --group gpu` (torch, lmdeploy, neucodec)
- Dev install: `uv sync --group dev` (includes pytest, transformers, librosa)

### Makefile (9492 bytes)
**Key Targets**:
- `make web` — Run vieneu-web UI
- `make test` — Run pytest suite
- `make lint` — Ruff + type checking
- `make install` — `uv sync` (core) or `uv sync --group gpu` (GPU)
- `make docker-build` — Build Docker image
- `make docker-run` — Run Docker container with GPU support

## Docker Deployment

**Dockerfile**: Multi-stage build
1. Base: Python 3.12 + CUDA 12.8
2. Install uv + dependencies ([gpu] group)
3. Copy model weights (optional mount)
4. Expose port 23333 (LMDeploy API)
5. Default: `vieneu serve` command

**docker-compose.yml**:
- GPU support via `runtime: nvidia`
- Volume mounts for HF cache & model outputs
- Environment variables: `CUDA_VISIBLE_DEVICES`, `HF_TOKEN`

## Code Statistics

| Component | Files | LOC | Purpose |
|-----------|-------|-----|---------|
| Core engines | 11 | 2,658 | Factory, base, v3turbo, v2, remote, xpu |
| V3 Turbo sub-engine | 7 | 1,095 | Transformer, ONNX, inference, serving |
| V3 Turbo serving | 4 | 376 | Batched pipeline, CUDA graph |
| Utilities | 3 | 410 | Phonemization, audio, URL extraction |
| Apps | 5 | 3,584 | Gradio web UI, streaming server, XPU UI |
| Tests | 7 | ~800 | Unit & integration tests |
| Fine-tuning | 7 | ~1,200 | LoRA training, merging, data prep |

**Total**: ~10,000 LOC (excluding models, data, tests)

---

**Last Updated**: 2026-06-25  
**Generated from**: repomix analysis + manual inspection
