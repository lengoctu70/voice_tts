---
title: "SRT to Audio Web UI feature"
description: "Paste SRT subtitles into a dedicated Web UI tab and generate one WAV whose speech aligns with the SRT timeline."
status: completed
priority: P2
branch: ""
tags: ["web-ui", "tts", "srt", "tdd"]
blockedBy: []
blocks: []
created: "2026-06-25T02:10:18.881Z"
createdBy: "ck:plan"
source: skill
---

# SRT to Audio Web UI feature

## Overview

Add an "SRT → Audio" tab to the existing Gradio Web UI (`apps/gradio_main.py`). The user pastes SRT subtitle text; the app reuses the currently loaded model and selected voice/temperature/chunk settings to synthesize each subtitle block, then assembles one WAV whose speech respects each subtitle's `start` timestamp.

Per-block audio is silence-trimmed, then matched to the subtitle's `[start, end]` window using three rules in order:
1. shorter → pad with trailing silence,
2. longer but within ≤1.25× speed-up → time-stretch to fit,
3. longer than 1.25× speed-up → hard-cut overflow.

Fully silent generated audio for a block with text triggers up to 2 retries with identical settings; persistent silence fails the job and identifies the offending block. SRT is validated before any TTS call.

Single-speaker only in v1. Uses the engine's active sample rate.

## Goals

- Timeline accuracy: each subtitle block starts at its SRT `start`.
- Zero regression on existing single-speaker and conversation flows.
- Concise per-chunk diagnostics surfaced in the UI status area.
- Pure-Python audio post-processing (no new heavy deps); pull in `librosa` only for time-stretch, falling back to a phase vocoder over `scipy.signal` if `librosa` is not desired.

## Non-Goals

- Multi-speaker SRT parsing or auto-detect speakers per block.
- Separate per-chunk downloadable files.
- Streaming the assembled WAV during generation.
- Changes to existing tabs' behavior.

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [SRT parser & validator](./phase-01-srt-parser-validator.md) | Completed |
| 2 | [Audio post-processing utilities](./phase-02-audio-post-processing-utilities.md) | Completed |
| 3 | [SRT-to-audio pipeline orchestrator](./phase-03-srt-to-audio-pipeline-orchestrator.md) | Completed |
| 4 | [Gradio UI tab integration](./phase-04-gradio-ui-tab-integration.md) | Completed |
| 5 | [Integration tests & docs](./phase-05-integration-tests-docs.md) | Completed |

Each phase follows the project TDD convention: write failing pytest cases first under `tests/`, then implement until green. Phases 1–3 must reach green before Phase 4 wires UI to the orchestrator.

## Architecture

```
apps/gradio_main.py                  ← new "SRT → Audio" tab + click handler
└── calls vieneu_utils.srt_to_audio.synthesize_srt(...)
        ├── vieneu_utils.srt_parser.parse_srt(text)    → list[SrtBlock]
        ├── vieneu_utils.srt_parser.validate_blocks(blocks)
        ├── (per block) tts.infer / tts.infer_stream   ← reuses loaded model
        └── vieneu_utils.srt_audio_ops:
            ├── trim_edge_silence(wav, sr)
            ├── fit_to_window(wav, target_dur, sr, max_speedup=1.25)
            │     → returns (wav, applied_speedup, was_cut)
            └── assemble_timeline(blocks_with_wavs, sr) → np.ndarray
```

Data shapes:
- `SrtBlock`: `{index:int, start_s:float, end_s:float, text:str}`.
- `ChunkLog`: `{index, start_s, end_s, retries, trim_lead_ms, trim_trail_ms, pad_ms, speedup, cut_ms, status}` where `status ∈ {ok, retried, cut, failed}`.

## Acceptance Criteria

Mirrors the task spec — verified by integration test in Phase 5 driving a small fixture SRT through the pipeline with a stubbed TTS that returns deterministic waveforms covering each branch (short / borderline / cut / silent-retry / silent-fail / invalid SRT).

- A valid SRT yields one WAV; existing flows untouched.
- Short chunks pad; medium chunks speed up only as needed (≤1.25×); long chunks are cut and the log reports it.
- Silent chunk retries up to 2× then fails with the offending block surfaced.
- Invalid SRT (malformed timestamps, missing text, wrong order, overlapping ranges, zero/negative duration) fails before any TTS call with a clear validation message.
- Per-chunk log is concise (one line per block) and includes retry count, trim, pad, speed-up, cut, failure reason when relevant.

## Risks

- **Time-stretch quality** at 1.25×: phase vocoder may introduce artifacts. Mitigation: cap is 1.25×, well below audible degradation for prosody-preserving stretchers like `librosa.effects.time_stretch`.
- **Engine sample-rate drift** across backbones (24 kHz vs 48 kHz): always read `getattr(tts, "sample_rate", ...)` at call time and use it for all silence/padding arrays.
- **Silence detection thresholds**: tuning RMS/peak thresholds for "fully silent" must avoid false positives on quiet speech. Use a conservative threshold (peak < 1e-3 and RMS < 1e-4 over whole clip) and verify against a known-good fixture in Phase 2 tests.
- **UI thread blocking**: long SRTs may take minutes. Use a generator handler (yield status) like the existing `synthesize_speech` flow.

## Dependencies

No cross-plan dependencies. Touches:
- `apps/gradio_main.py` (additive — new tab, new handler, no edits to existing handlers).
- `apps/ui_constants.py` (optional: default SRT placeholder).
- New module `src/vieneu_utils/srt_parser.py`.
- New module `src/vieneu_utils/srt_audio_ops.py`.
- New module `src/vieneu_utils/srt_to_audio.py`.
- New tests under `tests/`.

## Open Questions

- Confirm placement of new helpers under `src/vieneu_utils/` (vs `src/vieneu/`). Defaulting to `vieneu_utils` because they are stateless utilities and tests already live alongside `vieneu_utils.core_utils`.
- Confirm whether `librosa` is acceptable as a new dependency for time-stretch. If not, Phase 2 implements a small WSOLA fallback.
