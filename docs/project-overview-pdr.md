# VieNeu-TTS: Project Overview & Product Development Requirements

## Product Overview

**VieNeu-TTS** is an advanced, on-device Vietnamese text-to-speech engine with bilingual (Vi-En) code-switching, instant voice cloning, and conversation support. It targets researchers, developers, and content creators who need high-quality Vietnamese speech synthesis without cloud dependencies.

### Core Value Proposition
- **On-device inference**: Full offline capability; no internet required after model download
- **High-fidelity audio**: 48 kHz v3 Turbo (preview) and 24 kHz v2 multi-speaker support
- **Instant voice cloning**: Clone any voice in 3–5 seconds of reference audio
- **Bilingual code-switching**: English-Vietnamese seamless mixing via [sea-g2p](https://github.com/pnnbao97/sea-g2p)
- **Multiple backbone options**: From lightweight CPU-ONNX to GPU-accelerated PyTorch/LMDeploy
- **Production-ready**: Used in podcasts, dialogues, and AI assistants

## Target Users

1. **Researchers**: Phoneme-level control, emotion tagging (experimental), batched generation for dataset synthesis
2. **Developers**: SDK integration into apps/services; Python ≥3.10 requirement; optional GPU (CUDA 12.8+, Apple MPS)
3. **Podcasters & Content Creators**: Podcast/conversation mode with multi-speaker dialogue, voice cloning for character voices
4. **Edge/IoT Deployments**: CPU-ONNX path for latency-critical applications

## Product Architecture: Three Backbones

| Backbone | Codec | Sample Rate | Device | Use Case | Status |
|----------|-------|-------------|--------|----------|--------|
| **v3 Turbo** | MOSS-Audio-Tokenizer-Nano | 48 kHz | CPU (ONNX) / GPU (PyTorch) | Fast, batched, default voices | **Early Access** |
| **v2 Standard** | NeuCodec | 24 kHz | GPU (PyTorch) | Maximum quality, pod casting | **Stable** |
| **v1 Legacy** | NeuCodec | 24 kHz | GPU (PyTorch) | Stable baseline | **Stable** |

### VieNeu-TTS v3 Turbo (Early Access)
- **Architecture**: Custom transformer backbone with speaker tokens for default voices
- **Audio Quality**: 48 kHz high-fidelity waveforms
- **Features**:
  - Built-in default voices (Bình An, Ngọc Linh, Xuân Vĩnh, etc.) — no reference clip needed
  - Voice cloning from 3–5s reference audio
  - Emotion/non-verbal cues: `[cười]` (laugh), `[thở dài]` (sigh), `[hắng giọng]` (throat clear)
  - Batched generation (up to 32 samples per batch)
  - Multi-speaker conversation mode
- **Inference Path**: CPU → ONNX Runtime (torch-free); GPU → PyTorch (auto-detected)
- **Timeline**: Full v3 release planned for next few weeks

### VieNeu-TTS v2 Standard
- **Architecture**: Bilingual transformer with 10,000+ hours of training data
- **Audio Quality**: 24 kHz natural prosody
- **Features**:
  - En-Vi code-switching with [sea-g2p](https://github.com/pnnbao97/sea-g2p)
  - Podcast/conversation mode with automatic speaker detection
  - Zero-shot voice cloning from reference audio
  - Emotion modes: "natural" (conversational), "storytelling"
- **Inference Path**: GPU via LMDeploy (fast) or PyTorch (standard)

### VieNeu-TTS v1 (Legacy)
- Stable, production-proven baseline
- Vietnamese-only (no bilingual code-switching)
- 24 kHz audio quality

## Key Features & Capabilities

### 1. Voice Cloning (All Backbones)
- **Input**: 3–5 seconds of reference audio + target text
- **Method**: Reference encoder → speaker embedding → synthesis
- **Quality**: Zero-shot, works across backbones
- **API**: `tts.infer(text, ref_audio="path.wav", ref_text="reference text")`

### 2. Emotion & Non-verbal Cues (v3 Turbo, Experimental)
- **Syntax**: Embed tags directly in text: `[cười]`, `[thở dài]`, `[hắng giọng]`
- **Integration**: Via [sea-g2p](https://github.com/pnnbao97/sea-g2p) phonemizer
- **Usage**: `tts.infer("Chào bạn [cười]")`

### 3. Podcast/Conversation Mode (v2 & v3)
- **Multi-speaker**: Auto-detect or specify speaker lines (e.g., `[Speaker A]`, `[Speaker B]`)
- **Batching**: Entire dialogue batched for efficiency
- **Use Case**: Podcast generation, audiobook narration, dialogue scripts

### 4. Bilingual Code-switching (v2 & v3)
- **Languages**: Vietnamese + English
- **Phonemizer**: [sea-g2p](https://github.com/pnnbao97/sea-g2p) (Vietnamese-specific G2P)
- **Example**: "Xin chào Hello world" → seamless mixing

### 5. SRT → Audio (Single-speaker, Timeline-aligned)
- **Input**: An SRT subtitle block list pasted into the Web UI's **📜 SRT → Audio** tab.
- **Output**: A single WAV at the engine's active sample rate, with each subtitle block placed at its `start` timestamp.
- **Fitting rules per block** (in order):
  1. shorter than the window → pad with trailing silence;
  2. longer but within ≤1.25× speed-up → time-stretch to fit;
  3. longer beyond the cap → speed up at 1.25× then hard-cut the overflow.
- **Resilience**: Fully silent generation triggers up to 2 retries; persistent silence fails the job and identifies the offending block. SRT is validated before any TTS call.
- **Scope**: Single-speaker only in v1; reuses the loaded model + the selected voice / cloning reference / temperature / max-chars settings.

## System Requirements

### Minimum
- **Python**: ≥3.10 (3.13 tested)
- **CPU**: Any x86_64 / ARM64 (macOS M1+)
- **RAM**: 2GB for v3 Turbo ONNX, 4GB for GPU models

### GPU (Optional, for v2/v3 PyTorch paths)
- **NVIDIA**: CUDA ≥12.8, cuDNN 8, NVIDIA Container Toolkit (Docker)
- **Apple Silicon**: Metal Performance Shaders (MPS) via PyTorch
- **Intel**: Intel GPU drivers + torch.xpu (XPU backend)

### Dependency Groups
- **Core (torch-free)**: `sea-g2p`, `onnxruntime`, `numpy`, `soundfile`, `soxr`, `tokenizers`, `huggingface_hub`, `gradio`
- **[gpu]**: `torch`, `torchaudio`, `transformers`, `librosa`, `neucodec`, `lmdeploy` (Linux/Windows only)
- **[dev]**: `pytest`, `pytest-asyncio`, `librosa`, `transformers`, `neucodec`

## Use Cases

### Research & Development
- Dataset synthesis with emotion cues
- Phoneme-level control via sea-g2p
- Fine-tuning on custom data (LoRA adapters)
- Batched generation for benchmarking

### Production Deployment
- **Web UI**: Gradio app (`vieneu-web`) for real-time synthesis
- **Streaming Server**: FastAPI with bore tunnel support (`vieneu-stream`)
- **Remote API**: Docker LMDeploy server with client SDK
- **Embedded**: CPU-ONNX on resource-constrained devices

### Content Creation
- Podcasts with multi-speaker dialogue
- AI assistant voice personalization
- Audiobook narration with voice cloning

## Functional Requirements

### SDK Interface (Factory Pattern)
- **Single entry point**: `Vieneu(mode="v3turbo", **kwargs)`
- **Mode options**:
  - `"v3turbo"` (default) — v3 Turbo on CPU/GPU
  - `"standard"` — v1 Standard (GPU)
  - `"fast"` / `"gpu"` — v2 fast path (LMDeploy, GPU)
  - `"turbo"` / `"turbo_gpu"` — v2 Turbo (GGUF/PyTorch)
  - `"remote"` — Remote API server
  - `"xpu"` — Intel GPU

### Core Methods
- `infer(text, voice=None, ref_audio=None, ref_text=None, emotion="natural")` → Audio array
- `save(audio, path)` → Write WAV to disk
- `list_preset_voices()` → List built-in voices (v3 Turbo only)
- `get_preset_voice(voice_id)` → Voice metadata

### Configuration
- **config.yaml**: Backbone selection, streaming chunk sizes, text limits
- **Environment variables**: `HF_TOKEN` (Hugging Face access), `DEVICE` (CPU/GPU override)

## Non-functional Requirements

### Performance
- **v3 Turbo ONNX (CPU)**: ~100–200ms per second of audio (on modern CPU)
- **v3 Turbo PyTorch (GPU)**: <50ms per second of audio
- **v2 LMDeploy (GPU)**: <30ms per second of audio
- **Batched generation**: Linear scaling up to batch size 32

### Reliability
- **Model versioning**: Via Hugging Face releases; pinned in `pyproject.toml`
- **Fallback phonemization**: sea-g2p with manual punctuation normalization
- **Voice preset validation**: JSON schema for voices.json compatibility

### Maintainability
- **Modular design**: Engine backends (v3turbo.py, standard.py, fast.py) inheriting from base.py
- **Test coverage**: pytest + pytest-asyncio for core inference paths
- **Documentation**: README.md (primary), Deploy.md (Docker), custom model guides
- **Dependency management**: uv for reproducible lock files; optional [gpu] group for GPU-only deps

## Roadmap

| Phase | Milestone | Target | Status |
|-------|-----------|--------|--------|
| **Phase 1** | v2 Standard & Podcast Mode | Done | Released |
| **Phase 2** | v3 Turbo Architecture | Done | Early Access |
| **Phase 2.5** | v3 Full Release | Few weeks | In Progress |
| **Phase 3** | Mobile SDK (Android/iOS) | TBD | Planned |
| **Phase 4** | Advanced Emotion Control | TBD | Planned |
| **Phase 5** | More Default Voices | Ongoing | Released (v3 Turbo preview) |

## Acceptance Criteria

### v3 Turbo Early Access (Current)
- ✅ 48 kHz audio quality with MOSS codec
- ✅ Built-in speaker tokens (Bình An, Ngọc Linh, Xuân Vĩnh, etc.)
- ✅ Voice cloning from 3–5s reference
- ✅ Emotion tag support (experimental)
- ✅ Batched generation (up to 32 samples)
- ✅ Multi-speaker conversation mode
- ✅ ONNX runtime for CPU (torch-free)
- ✅ PyTorch engine for GPU (auto-detected)

### Full v3 Release (Next Phase)
- Finalized emotion control
- Expanded default voice library
- Streaming server support
- Stable API guarantees

## Scope & Non-Goals

### In Scope
- Vietnamese-English bilingual synthesis
- Voice cloning for any input voice
- Batched generation for efficiency
- CPU + GPU inference paths
- Docker deployment

### Out of Scope (v3 Phase)
- Real-time streaming (v2 only for now)
- Non-Vietnamese/English languages
- Mobile native SDKs (planned Phase 3)
- Web-based model training UI

## Security & Compliance

- **License**: Apache 2.0 (permissive, free to use)
- **Model Access**: All models via Hugging Face (public or gated)
- **Data Privacy**: Models run locally; no telemetry
- **Watermarking**: Optional (via [perth](https://github.com/pnnbao97/perth) library)

## Support & Communication

- **Issues**: GitHub repository
- **Discord**: Community support channel
- **Hugging Face**: Model cards and discussions
- **Documentation**: README + Deploy guides

---

**Last Updated**: 2026-06-25  
**Maintained by**: Phạm Nguyễn Ngọc Bảo
