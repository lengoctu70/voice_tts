# VieNeu-TTS System Architecture

## High-Level Layered Design

```
┌─────────────────────────────────────────────────┐
│  User Interfaces                                │
│  ├─ Gradio Web UI (vieneu-web)                  │
│  ├─ FastAPI Streaming Server (vieneu-stream)   │
│  └─ Remote API Client (Python SDK)              │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  Factory Dispatcher (factory.py)                │
│  Routes mode string → engine instance           │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  Abstract Base Class (base.py)                  │
│  ├─ Common voice management                     │
│  ├─ Preset voice loading                        │
│  ├─ Codec initialization                        │
│  └─ File I/O (save/load)                        │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  Engine Backends (Mode-Specific)                │
│  ├─ V3TurboVieNeuTTS (v3turbo) ← DEFAULT        │
│  ├─ VieNeuTTS (standard)       ← v1 GPU         │
│  ├─ FastVieNeuTTS (fast)       ← v2 LMDeploy    │
│  ├─ TurboVieNeuTTS (turbo)     ← v2 GGUF        │
│  ├─ RemoteVieNeuTTS (remote)   ← API client     │
│  └─ XPUVieNeuTTS (xpu)         ← Intel GPU      │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  Sub-engines & Utilities                        │
│  ├─ _v3_turbo_engine/         (V3 Turbo logic) │
│  │  ├─ modeling_v3_turbo.py   (Transformer)    │
│  │  ├─ inference_v3_turbo.py  (Forward pass)   │
│  │  └─ onnx_runtime_lite.py   (ONNX wrapper)   │
│  │                                              │
│  ├─ v3_turbo_serve/           (Batched serving)│
│  │  ├─ batched_backbone.py                     │
│  │  ├─ batched_acoustic.py                     │
│  │  └─ cudagraph.py            (GPU opt)       │
│  │                                              │
│  └─ vieneu_utils/             (Shared)         │
│     ├─ phonemize_text.py      (G2P + emotion)  │
│     ├─ core_utils.py          (Audio processing)│
│     └─ url_extract.py         (Podcast helper) │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│  External Dependencies                          │
│  ├─ sea-g2p                  (Phonemization)   │
│  ├─ ONNX Runtime             (CPU inference)   │
│  ├─ PyTorch + Transformers   (GPU inference)   │
│  ├─ neucodec / MOSS codec    (Audio decoding)  │
│  ├─ LMDeploy                 (Fast inference)  │
│  ├─ huggingface_hub          (Model download)  │
│  └─ Gradio / FastAPI         (Web frameworks)  │
└─────────────────────────────────────────────────┘
```

## Engine Selection Flow

### Factory Dispatch (factory.py:18–42)

```python
def Vieneu(mode="v3turbo", **kwargs):
    match mode:
        case "v3turbo":           # DEFAULT — CPU/GPU auto-detect
            return V3TurboVieNeuTTS(**kwargs)
        case "standard":          # v1 GPU
            return VieNeuTTS(**kwargs)
        case "fast" | "gpu":      # v2 LMDeploy (GPU)
            return FastVieNeuTTS(**kwargs)
        case "turbo":             # v2 GGUF (CPU/GPU)
            return TurboVieNeuTTS(**kwargs)
        case "remote" | "api":    # HTTP API client
            return RemoteVieNeuTTS(**kwargs)
        case "xpu":               # Intel GPU
            return XPUVieNeuTTS(**kwargs)
```

### Decision Tree: Which Backend?

```
Start: Vieneu(mode="...")
│
├─ "v3turbo" (DEFAULT)
│  ├─ Device detection
│  │  ├─ GPU available + PyTorch installed → PyTorch v3 Turbo
│  │  └─ CPU only (or torch unavailable) → ONNX v3 Turbo (torch-free)
│  │
│  └─ Inference
│     ├─ PyTorch: V3TurboTransformer.forward() → acoustic codes
│     │           → MOSS codec decode → waveform (48 kHz)
│     └─ ONNX:    onnx_runtime_lite.run() → acoustic codes
│                 → MOSS codec ONNX decode → waveform (48 kHz)
│
├─ "standard" (v1 GPU)
│  ├─ Check CUDA available
│  └─ Inference: VieNeuTTS.forward() → mel-spectrogram
│               → NeuCodec decode → waveform (24 kHz)
│
├─ "fast" or "gpu" (v2 LMDeploy)
│  ├─ Start LMDeploy engine
│  └─ Inference: Quantized backbone
│               → LMDeploy forward → acoustic codes → waveform (24 kHz)
│
├─ "turbo" (v2 GGUF)
│  ├─ Load GGUF quantized model
│  └─ Inference: llama.cpp → acoustic codes → waveform (24 kHz)
│
├─ "remote" or "api" (HTTP)
│  ├─ Connect to REMOTE_API_BASE (e.g., http://server:23333/v1)
│  └─ Inference: Client encodes ref audio locally
│               → POST /synthesize → server backbone → codes
│               → client decodes codes → waveform
│
└─ "xpu" (Intel GPU)
   ├─ Check torch.xpu available
   └─ Inference: Intel GPU via torch.xpu ↔ PyTorch backbone
```

## V3 Turbo Engine Architecture (Default)

### Initialization Path (v3turbo.py:25–80)

```python
V3TurboVieNeuTTS(
    backbone_repo="pnnbao-ump/VieNeu-TTS-v3-Turbo",
    moss_tokenizer="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
    device="auto",    # auto → CPU or cuda
    dtype="auto",     # auto → float32
    backend="auto",   # auto → ONNX (CPU) or PyTorch (GPU)
)
```

**Device Detection Logic**:
```python
if device == "auto":
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except:
        device = "cpu"  # Torch-free path (CPU + ONNX)
```

**Backend Selection Logic**:
```python
if backend == "auto":
    if device == "cpu":
        backend = "onnx"      # CPU → ONNX Runtime (fast, torch-free)
    elif device == "cuda":
        backend = "pytorch"   # GPU → PyTorch (GPU-optimized)
```

### Inference Pipeline (v3turbo.py:infer)

```
Input Text
    ↓
[1] Text Normalization (sea-g2p Normalizer)
    ├─ Punctuation cleanup
    └─ Diacritic normalization
    ↓
[2] Phonemization + Emotion Extraction (phonemize_text_with_emotions)
    ├─ sea-g2p phonemizer → phoneme tokens
    ├─ Extract emotion tags: [cười], [thở dài], [hắng giọng]
    └─ Output: (phoneme_ids, emotion_ids)
    ↓
[3] Text Chunking (max 256 chars per chunk)
    ├─ Split long text into manageable pieces
    └─ Process sequentially or batched
    ↓
[4] Model Inference (device-specific)
    ├─ PyTorch Path:
    │  ├─ V3TurboTransformer.forward(
    │  │     input_ids=phoneme_ids,
    │  │     speaker_ids=speaker_token,
    │  │     emotion_ids=emotion_ids
    │  │  ) → acoustic_codes [shape: (seq_len, codec_dim)]
    │  │
    │  └─ Optional: CUDA Graph replay (cudagraph.py)
    │
    └─ ONNX Path:
       ├─ onnx_runtime_lite.run(
       │     inputs={
       │         "input_ids": phoneme_ids,
       │         "speaker_ids": speaker_token,
       │         "emotion_ids": emotion_ids
       │     }
       │  ) → acoustic_codes
       │
       └─ Auto-selects best ONNX provider (CPU, TensorRT if available)
    ↓
[5] Acoustic Codec Decoding (MOSS-Audio-Tokenizer-Nano)
    ├─ codes → latent vectors
    └─ decoder → waveform [sample_rate=48kHz, dtype=float32]
    ↓
[6] Audio Post-processing
    ├─ Optional loudness normalization
    ├─ Chunk joining (with overlap handling)
    └─ Output: numpy.ndarray [shape: (samples,), dtype=float32, sr=48kHz]
```

### V3 Turbo Transformer Architecture

**File**: `src/vieneu/_v3_turbo_engine/modeling_v3_turbo.py` (248 LOC)

```
Input: Phoneme IDs (batch, seq_len)
    ↓
[Embedding Layer]
├─ Phoneme embedding: (seq_len) → (seq_len, 768)
├─ Speaker embedding: speaker_token → (1, 768)
└─ Emotion embedding: emotion_id → (1, 768)
    ↓
[Stack of Transformer Layers (default: 8)]
├─ Layer 1-8: (seq_len, 768)
│  ├─ MultiHeadSelfAttention(heads=8, hidden=768)
│  │  └─ Causal masking (AR generation)
│  ├─ FeedForward(hidden=768, ffn=3072)
│  ├─ LayerNorm + residual connections
│  └─ Dropout (0.1)
└─ Output: (seq_len, 768)
    ↓
[Output Projection]
└─ Linear(768 → codec_dim) → acoustic codes
    ↓
Output: Acoustic codes [shape: (seq_len, codec_dim)]
```

**Key Design Choices**:
- **Causal masking**: Auto-regressive generation (left-to-right)
- **Speaker token**: Learnable embedding for each default voice (no reference needed)
- **Emotion embedding**: Discretized emotion tags mapped to learned vectors
- **No cross-attention**: v3 uses concatenated embeddings (simpler, faster)

### ONNX Runtime Selection (onnx_runtime_lite.py)

**Provider Priority** (fallback order):
```python
providers = [
    ("CUDAExecutionProvider", {"device_id": 0}),     # NVIDIA GPU
    ("TensorrtExecutionProvider", {"device_id": 0}), # TensorRT (fastest)
    ("CPUExecutionProvider", {})                      # CPU fallback
]
session = ort.InferenceSession(model_path, providers=providers)
```

**Benefits**:
- ONNX on CPU: 50–100ms per second of audio (modern CPU)
- ONNX on GPU: <20ms per second of audio
- Torch-free: Can run on minimal systems (2GB RAM, no CUDA)

## Batched Serving Pipeline (v3_turbo_serve/)

### Purpose
Maximize throughput by processing multiple synthesis requests in parallel.

### Architecture (engine.py:137 LOC)

```
[Request Queue]
└─ Multiple texts, speakers, emotions
    ↓
[Batch Grouping]
├─ Group by speaker (optional)
└─ Limit batch size ≤ 32
    ↓
[Batched Phonemization]
├─ Phonemize all texts in parallel
└─ Pad to max sequence length
    ↓
[Batched Backbone Inference] (batched_backbone.py)
├─ V3TurboTransformer.forward(
│     input_ids=[all phoneme sequences, padded],
│     speaker_ids=[all speaker tokens]
│  ) → batch of acoustic codes
└─ Output shape: (batch_size, max_seq_len, codec_dim)
    ↓
[Batched Acoustic Decoding] (batched_acoustic.py)
├─ MOSS codec decoder on GPU (parallel decode)
└─ Output: (batch_size, num_samples) waveforms [48kHz]
    ↓
[CUDA Graph Optimization] (cudagraph.py, GPU only)
├─ First run: torch.cuda.make_graphed_callables()
└─ Replay: ~20–30% latency reduction
    ↓
[Response Streaming]
└─ Stream waveforms to client as they complete
```

### Performance Scaling

| Batch Size | Texts/sec (GPU) | Latency/sample | Notes |
|------------|-----------------|---|---|
| 1 | ~5–10 | High overhead | Single request |
| 8 | ~40–50 | Amortized | Sweet spot |
| 32 | ~100–120 | Low | Max throughput |

**Batching is cumulative**: Each additional text adds ~10–20ms overhead, regardless of length.

## Phonemization Pipeline (sea-g2p Integration)

### Text → Phonemes Workflow

**File**: `src/vieneu_utils/phonemize_text.py` (148 LOC)

```python
def phonemize_text_with_emotions(text, lang="vi"):
    """
    Input: "Chào bạn [cười]"
    ↓
    [1] Extract emotion tags
        └─ Find [tag] patterns → {tag_name, position}
    ↓
    [2] Remove emotion tags from text
        └─ "Chào bạn"
    ↓
    [3] Phonemize (sea-g2p)
        └─ V → [v], i → [i], etc.
        └─ Output: "v1 ao1 b a: n"
    ↓
    [4] Insert emotion tags back
        └─ Map positions to phoneme sequence
    ↓
    Output: (phoneme_ids, emotion_ids)
    """
```

### Supported Languages

- **Vietnamese** (`lang="vi"`): Via sea-g2p Vietnamese lexicon
- **English** (`lang="en"`): Via sea-g2p English lexicon
- **Code-switching**: sea-g2p auto-detects mixed text

**Emotion Tags** (Experimental, v3 Turbo):
| Tag | Phoneme | Meaning |
|-----|---------|---------|
| `[cười]` | CUU1 | Laughter |
| `[thở dài]` | THO2 DAI4 | Sigh |
| `[hắng giọng]` | HANG2 GIONG | Throat clear |
| `[tức]` | TUC2 | Anger |

### Fallback Handling

```python
try:
    phonemes = normalizer(text)  # sea-g2p
except Exception as e:
    logger.warning(f"sea-g2p failed: {e}")
    phonemes = text.split()  # Fallback: whitespace split
```

## Audio I/O & Codec Architecture

### Loading Reference Audio (base.py:_load_ref_mono)

**Priority order**:
1. **librosa** (if [gpu] installed)
   ```python
   import librosa
   wav, _ = librosa.load(path, sr=target_sr, mono=True)
   ```

2. **soundfile + soxr** (core dependencies, torch-free)
   ```python
   import soundfile as sf
   import soxr
   wav, sr = sf.read(path, dtype="float32")
   wav = soxr.resample(wav, sr, target_sr)
   ```

3. **Fallback**: Error if none available

### Codec Architecture

#### V3 Turbo: MOSS-Audio-Tokenizer-Nano

- **Encoder**: Quantize waveform → discrete tokens
- **Decoder**: Reconstruct waveform from tokens
- **Format**: ONNX (CPU/GPU) or PyTorch
- **Sample rate**: 48 kHz (16-bit)
- **Latency**: <10ms per second of audio

#### V1/V2: NeuCodec

- **Type**: Learned audio codec (neural vocoder)
- **Format**: PyTorch (GPU only) or ONNX (CPU, experimental)
- **Sample rate**: 24 kHz (16-bit)
- **Codec variant**: "NeuCodec (Distill)" (lightweight)

### Waveform Output Format

All backends return:
- **dtype**: float32 (range -1.0 to 1.0)
- **shape**: (num_samples,) mono
- **sample_rate**: 48 kHz (v3) or 24 kHz (v1/v2)

**Saving to disk** (base.py:save):
```python
import soundfile as sf
sf.write(path, audio, samplerate=self.sample_rate, subtype='PCM_16')
```

## Voice Cloning Architecture

### Reference Audio Encoding

**Pipeline**:
```
Reference Audio (3–5s)
    ↓
[Load & Normalize]
├─ Load as float32 mono
├─ Resample to 48 kHz (v3) or 24 kHz (v1/v2)
└─ Normalize loudness
    ↓
[Reference Encoder]
├─ Forward pass through trained encoder
├─ Extract speaker embedding (512-dim vector)
└─ Cache for later synthesis
    ↓
[Synthesis with Cloned Voice]
├─ Use speaker embedding instead of preset token
├─ Feed to transformer backbone
└─ Generate audio in cloned voice
```

**Zero-shot capability**: No fine-tuning required; works with any reference voice.

### Remote Mode Voice Cloning

```
Client Side:
  reference_audio.wav
      ↓
  [Client-side encoder]
  → speaker_embedding (vector)
  → HTTP POST /synthesize
      {text, speaker_embedding, ref_text}
      ↓
  
Server Side:
  [Backbone inference]
  → acoustic_codes
  
  [Client-side decode]
  ← acoustic_codes
  → waveform.wav
```

**Benefit**: Reference never leaves client; only 512-dim vector sent over network.

## Deployment Topologies

### 1. Local Gradio (vieneu-web)

```
User Browser (http://localhost:7860)
    ↓ HTTP
Gradio Web Server (gradio_main.py)
    ↓ (synchronous)
Vieneu SDK (Vieneu(mode="v3turbo"))
    ├─ CPU ONNX path
    └─ GPU PyTorch path
    ↓
Output: WAV file
    ↓ Download
User
```

**Latency**: ~100–500ms per sentence (CPU dependent)

### 2. FastAPI Streaming Server (vieneu-stream)

```
Client (vieneu SDK with mode="remote")
    ↓ HTTP/1.1 Streaming
FastAPI Server (web_stream.py:8000)
    ↓ async
Vieneu SDK (Vieneu(mode="v3turbo"))
    ├─ Batched inference
    └─ Streaming response
    ↓
Audio chunks
    ↓ Stream
Client (real-time playback)
```

**Latency**: <100ms time-to-first-chunk (batched)

### 3. Docker GPU (LMDeploy Server)

```
Docker Container (port 23333)
    ├─ LMDeploy vLLM backend (quantized v2 model)
    ├─ CUDA 12.8 + GPU
    └─ Bore public tunnel
    ↓ HTTP
Remote Client (vieneu SDK mode="remote")
    ├─ Lightweight (no model loaded)
    └─ Encodes ref audio locally
    ↓
Synthesis result (24 kHz audio)
```

**Use case**: Web apps, Colab notebooks, edge devices

### 4. Embedded/Edge (CPU ONNX)

```
IoT Device / Mobile (Python ≥3.10)
    ↓
pip install vieneu (torch-free)
    ↓
Vieneu(mode="v3turbo")
    ├─ ONNX Runtime (2–5MB)
    ├─ CPU inference
    └─ 48 kHz audio
    ↓
Local synthesis (latency ~200–500ms)
```

**Constraint**: Minimal storage/memory footprint

## Configuration & Runtime State

### Global Configuration (config.yaml)

```yaml
text_settings:
  max_chars_per_chunk: 256        # Synthesis chunk size
  max_total_chars_streaming: 3000 # Max text for streaming

backbone_configs:
  "VieNeu-TTS-v3-Turbo (Thử nghiệm)":
    repo: pnnbao-ump/VieNeu-TTS-v3-Turbo
    supports_streaming: false
    description: "48kHz, voice cloning, emotion cues"

codec_configs:
  "NeuCodec (Distill)":
    repo: neuphonic/distill-neucodec
    use_preencoded: false
```

**Used by**: Gradio web UI for dropdown selection & streaming config

### Runtime State Management (base.py)

```python
class BaseVieneuTTS:
    def __init__(self):
        self.sample_rate = 24_000           # Audio sample rate
        self.max_context = 2048             # Max token sequence
        self.hop_length = 480               # Codec hop size
        
        self._preset_voices = {}            # {voice_id: voice_data}
        self._ref_phoneme_cache = {}        # {ref_text: phoneme_ids}
        self.normalizer = Normalizer()      # sea-g2p instance
        self.watermarker = None             # perth watermark (optional)
        self.codec = None                   # Loaded codec instance
```

**Lifecycle**:
1. Instantiate: `tts = Vieneu(mode="v3turbo")`
2. Load models: Hugging Face auto-download to `~/.cache/huggingface/`
3. Inference: `audio = tts.infer(text)`
4. Save: `tts.save(audio, path)`
5. Cleanup: automatic (no explicit context manager needed)

## Streaming Architecture (Experimental, v3 Turbo)

### Streaming Inference Path

```
[Long text]
    ↓
[Chunked streaming]
├─ Chunk 1 (256 chars)
│   ↓ infer + stream
│   → audio chunk 1
│
├─ Chunk 2 (256 chars)
│   ↓ infer + stream
│   → audio chunk 2
│
└─ Chunk N
    ↓ infer + stream
    → audio chunk N
    ↓
[Join chunks with overlap]
└─ Smooth transitions
    ↓
[Stream to client in real-time]
```

**Parameters** (base.py:streaming_*):
- `streaming_overlap_frames = 1` — Overlap for smoothing
- `streaming_frames_per_chunk = 50` — Frames per network send
- `streaming_stride_samples = hop_length * frames_per_chunk`

**Latency**: ~50–100ms time-to-first-chunk

---

**Last Updated**: 2026-06-25  
**Architecture Version**: 3.0.5
