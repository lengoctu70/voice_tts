# VieNeu-TTS Project Roadmap

## Release Timeline & Status

### Phase 1: VieNeu-TTS v2 Standard & Podcast Mode

**Status**: ✅ **RELEASED**

**Timeline**: Released

**Deliverables**:
- ✅ VieNeu-TTS v2 with 10,000+ hours bilingual training (En-Vi)
- ✅ High-fidelity 24 kHz audio synthesis
- ✅ Podcast/Conversation mode with automatic speaker detection
- ✅ Zero-shot voice cloning (3–5s reference audio)
- ✅ En-Vi code-switching via [sea-g2p](https://github.com/pnnbao97/sea-g2p)
- ✅ Emotion modes: "natural" (conversational) and "storytelling"
- ✅ GPU optimization via LMDeploy
- ✅ Docker deployment with bore tunnel support
- ✅ Remote API client (lightweight, no GPU on client)

**Key Features**:
- Automatic character/speaker detection for podcasts
- Multi-speaker dialogue synthesis
- Seamless English-Vietnamese transitions
- Production-ready inference latency

**Backward Compatibility**: v1 (legacy) still available as `mode="standard"`

---

### Phase 2: VieNeu-TTS v3 Turbo Architecture

**Status**: 🟡 **EARLY ACCESS / PREVIEW**

**Timeline**: Current (preview released, full release in few weeks)

**Deliverables**:
- ✅ Brand-new 48 kHz architecture (designed from scratch by Phạm Nguyễn Ngọc Bảo)
- ✅ MOSS-Audio-Tokenizer-Nano codec (higher quality than v2's NeuCodec)
- ✅ Built-in default speaker tokens (no reference clip required)
  - Bình An (male, neutral)
  - Ngọc Linh (female, neutral)
  - Xuân Vĩnh (male, Southern accent)
  - And others (expanding)
- ✅ Voice cloning from 3–5s reference audio
- ✅ Emotion/non-verbal cue support (experimental)
  - `[cười]` (laugh)
  - `[thở dài]` (sigh)
  - `[hắng giọng]` (throat clear)
  - Additional cues planned
- ✅ Batched generation (batch size up to 32)
- ✅ Multi-speaker conversation mode (batch entire dialogue)
- ✅ CPU inference via ONNX Runtime (torch-free)
- ✅ GPU inference via PyTorch (auto-detected)
- ✅ Transformer-based backbone with speaker token embeddings

**Current Status** (Preview):
- Core v3 Turbo engine complete and tested
- Available in Web UI (`vieneu-web`) under backbone selector
- Available in SDK: `Vieneu(mode="v3turbo")`
- Emotion cues working (experimental, further refinement ongoing)
- Batched serving pipeline operational

**Planned for Full Release** (next few weeks):
- [ ] Finalize emotion control quality & stability
- [ ] Expand default speaker library (10+ voices)
- [ ] Streaming server support (vieneu-stream compatibility)
- [ ] Comprehensive testing across hardware platforms
- [ ] Production API stability guarantees
- [ ] Detailed emotion documentation & guidelines

**Key Advantages**:
- **Speed**: 48 kHz at ~100–200ms/sec on CPU (ONNX)
- **Quality**: Higher fidelity than v2 (48 kHz vs 24 kHz)
- **Flexibility**: Instant voice cloning + built-in voices
- **Accessibility**: CPU-ONNX requires no GPU or torch
- **Batching**: Ideal for content creators & batch processing

---

### Phase 2.5: Continued v2 Production Support

**Status**: 🟢 **ONGOING**

**Timeline**: Ongoing (v2 not deprecated)

**Commitments**:
- ✅ v2 remains stable production backbone
- ✅ Bug fixes & performance optimizations continue
- ✅ Community support & issue resolution
- ✅ Docker image updates for new CUDA versions
- ✅ Fine-tuning & LoRA training support (v2 compatible)

**When to use v2 instead of v3**:
- Maximum audio quality requirement (established baseline)
- Existing production pipeline (v2 tested extensively)
- Specific LMDeploy optimization needs (GPU deployment)

---

### Phase 3: Mobile SDK & Cross-Platform Support

**Status**: 📋 **PLANNED**

**Timeline**: TBD (estimated Q3 2026+)

**Proposed Deliverables**:
- [ ] Official Android SDK (Java/Kotlin)
- [ ] Official iOS SDK (Swift)
- [ ] React Native wrapper
- [ ] Flutter plugin
- [ ] WASM version for browser inference (experimental)
- [ ] On-device model quantization (4-bit / 8-bit)

**Approach**:
- Use ONNX Runtime Mobile for CPU inference
- Minimal model size (<100MB for v3 Turbo)
- Streaming audio output to system speakers
- Voice cloning via on-device reference encoding

**Barriers**:
- Model size optimization (currently ~500MB for full v3)
- Testing across diverse Android/iOS devices
- Platform-specific audio APIs (ALsa, AudioKit, etc.)

---

### Phase 4: Advanced Emotion Control & Non-verbal Cues

**Status**: 📋 **PLANNED**

**Timeline**: TBD (estimated Q4 2026)

**Proposed Features**:
- [ ] Fine-grained emotion parameter control (0–100 scale)
- [ ] Additional non-verbal cues
  - `[uống nước]` (drinking water sound)
  - `[ngoại]` (background noise)
  - `[bước chân]` (footsteps)
  - `[cof]` (cough)
  - `[khịt mũi]` (sniff)
- [ ] Emotion blending (mix two emotions with ratio)
- [ ] Speaker-specific emotion styles (emotion variation per voice)
- [ ] Prosody control (speed, pitch, emphasis)
  - `[nhanh]` (fast)
  - `[chậm]` (slow)
  - `[nhấn]` (emphasize)

**Current Limitation**: Emotion tags are discrete (on/off); phase 4 will add continuous control

**Research**: Community feedback & emotion annotation (will collect samples)

---

### Phase 5: Expanded Default Voice Library

**Status**: 🟡 **ONGOING (PREVIEW)**

**Timeline**: Gradual rollout (new voices added as they become available)

**Current Voices** (v3 Turbo preview):
- Bình An (nam, Miền Bắc)
- Ngọc Linh (nữ, Miền Bắc)
- Xuân Vĩnh (nam, Miền Nam)
- And several others

**Planned Additions**:
- [ ] More regional accents (Miền Trung, Tây Nguyên)
- [ ] More gender/age variety
- [ ] Child voices (3–12 years old, for storytelling)
- [ ] Elderly voices (for diverse narration)
- [ ] Whisper/intimate voice (ASMR-like)
- [ ] Dramatic/theatrical voices (for audiobooks)

**Selection Criteria**:
- Community voting (via Discord/GitHub)
- Speaker diversity (gender, age, region, emotion)
- Quality & distinctiveness
- License & consent from speakers

---

## Ongoing Initiatives

### A. Fine-tuning & LoRA Adapters

**Status**: 🟢 **ACTIVE**

- `finetune/train.py` — Train LoRA adapters on custom data
- `finetune/merge_lora.py` — Merge adapters into standalone weights
- `finetune/create_voices_json.py` — Create custom voice presets
- Support for v1, v2, and v3 Turbo models

**Use Cases**:
- Custom voice training on proprietary data
- Fine-tuning for specific accents or speech patterns
- Brand-voice synthesis (e.g., AI assistant mimicking CEO)

### B. Community Support & Feedback

**Status**: 🟢 **ONGOING**

- **Discord Community**: [Join Us](https://discord.gg/yJt8kzjzWZ)
- **GitHub Issues**: Feature requests, bug reports
- **Hugging Face**: Model discussions & feedback
- **Facebook**: Project updates & announcements

**Recent Community Contributions**:
- Emotion cue suggestions (from users)
- Performance optimization feedback (ONNX vs PyTorch)
- Deploy guidance refinements

### C. Performance & Optimization

**Status**: 🟢 **ONGOING**

- CUDA Graph optimization (v3 Turbo serving)
- ONNX quantization research (for edge devices)
- Batching throughput improvements
- Streaming latency reduction

**Current Benchmarks** (v3 Turbo on typical hardware):
- CPU (Intel i7, ONNX): ~150ms per second of audio
- CPU (M1 Mac, ONNX): ~100ms per second of audio
- GPU (RTX 3060, PyTorch): <50ms per second of audio
- GPU (RTX 4090, PyTorch): <20ms per second of audio

**Optimization Target**: <100ms/sec on modern CPUs (torch-free)

### D. Documentation & Examples

**Status**: 🟢 **ONGOING**

- README.md — Primary reference (comprehensive)
- Deploy.md — Docker & remote deployment guide
- CUSTOM_MODEL_USAGE.md — Fine-tuning instructions
- examples/ — Python SDK usage patterns
- Gradio UI — Interactive documentation

**Planned Docs**:
- [ ] Detailed emotion cue guide (once finalized)
- [ ] Performance tuning guide (batching, streaming)
- [ ] Troubleshooting FAQ
- [ ] Architecture deep-dive (technical)

---

## Feature Request Tracking

### Popular Community Requests

1. **Real-time Streaming** (High priority)
   - Partial support in v3 preview
   - Full support planned for v3 full release

2. **Multi-language Support** (Evaluated)
   - Currently: Vietnamese + English code-switching
   - Future: Mandarin, Thai, other Southeast Asian languages
   - Blocker: Need training data & native speakers

3. **Voice Conversion** (Investigated)
   - Convert one voice to another (A's words → B's voice)
   - Requires separate encoder; not in current roadmap
   - Community interest: collect feedback

4. **Music Generation** (Out of Scope)
   - VieNeu-TTS is speech-focused
   - Suggest: Try Jukebox, MusicLM (different tools)

5. **Web-based Training UI** (Out of Scope)
   - Too complex; CLI-based fine-tuning (finetune/) sufficient
   - Alternative: User uploads data → cloud training service (future startup idea?)

---

## Non-Goals (Intentionally Out of Scope)

### Scope Boundaries
- **Mobile native** (Phase 3, not v3 Turbo preview)
- **Real-time streaming** (Experimental in v3 preview, full in v3 release)
- **Non-Vietnamese languages** (English code-switching is limit for v3)
- **Voice conversion** (Requires separate architecture)
- **Music generation** (Different problem space)
- **WebRTC deployment** (Browser-based, not planned)
- **Commercial licensing** (Apache 2.0 free license is policy)

---

## Maintenance & Support Horizon

### Version Support Policy

| Version | Status | Support Until | Notes |
|---------|--------|---|---|
| **v1 (Legacy)** | ⚠️ Legacy | 2026-12-31 | Bug fixes only; no new features |
| **v2 (Standard)** | 🟢 Stable | Ongoing | Production-grade; community support |
| **v3 Turbo (Preview)** | 🟡 Early Access | 2026-08-31 | Full release planned; active development |
| **v3 Turbo (Full)** | 🟢 Stable (future) | TBD | After full release in weeks |

### Security & Vulnerability Handling
- Report security issues: pnnbao@gmail.com
- Responsible disclosure: 90-day embargo before public announcement
- Regular dependency updates via uv.lock

---

## Success Metrics

### Adoption KPIs
- PyPI downloads: >10k/month (current target)
- GitHub stars: >1k (tracking growth)
- Community Discord members: >500
- Production deployments: >100 (estimated)

### Quality KPIs
- Test coverage: >80% of core inference paths
- Bug resolution time: <2 weeks
- Performance regression: <5% tolerance between releases
- User satisfaction: >4.0/5.0 on surveys

### User Feedback Metrics
- Feature request volume: Prioritize top 10 annually
- Community issues: <10% unresolved after 1 month
- Documentation clarity: Reduce "FAQ" issues by 30% YoY

---

## How to Contribute to the Roadmap

### Suggestions & Voting
1. **GitHub Discussions**: Propose new features
2. **Discord**: Real-time feedback from community
3. **GitHub Issues**: Detailed feature requests with use cases
4. **Vote**: Reactions (👍) on issues influence priority

### Getting Involved
- **Code contributions**: Fork → PR with test coverage
- **Fine-tuning**: Share custom voice datasets (with permission)
- **Documentation**: Translate, expand examples, improve clarity
- **Testing**: Report edge cases, performance issues on your hardware

### Attribution
- Contributors get credit in CHANGELOG & GitHub contributors page
- Regular contributors offered maintainer role (optional)

---

## Key Dependencies & Future Monitoring

### Critical External Dependencies
- **sea-g2p**: Phonemization (maintained by pnnbao97; stable)
- **ONNX Runtime**: CPU inference (Microsoft; widely used; stable)
- **PyTorch**: GPU inference (Meta; leading framework; stable)
- **Hugging Face Hub**: Model distribution (no planned changes)

### Monitoring for Changes
- New CUDA versions (12.9+?) → test compatibility
- PyTorch major versions → benchmark performance
- sea-g2p updates → phonemization accuracy check
- Community fork activity (watch for derivative projects)

---

**Last Updated**: 2026-06-25  
**Maintained by**: Phạm Nguyễn Ngọc Bảo  
**Questions?**: [Discord Community](https://discord.gg/yJt8kzjzWZ)
