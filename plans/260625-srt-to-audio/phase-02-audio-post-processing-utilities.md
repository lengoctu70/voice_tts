---
phase: 2
title: "Audio post-processing utilities"
status: completed
effort: "M"
---

# Phase 2: Audio post-processing utilities

## Overview

Pure-NumPy helpers that turn a raw mono float32 waveform into a timeline-friendly chunk: silence trim at edges, fit-to-window with optional speed-up and hard-cut, silence detection, and final timeline assembly. No TTS or Gradio imports here — keeps Phase 3 testable with stubbed audio.

## Files

- Create `src/vieneu_utils/srt_audio_ops.py`.
- Create `tests/test_srt_audio_ops.py`.

## TDD — Tests first

`tests/test_srt_audio_ops.py`:

1. `is_silent` — returns True for all-zero buffer, False for a 1 kHz sine, True for buffer with peak < `1e-3`.
2. `trim_edge_silence` — synthetic signal `[silence | tone | silence | tone | silence]` keeps inner gap, trims outer; reports trimmed lead/trail in samples.
3. `fit_to_window` — short input → pads with trailing zeros, returns `applied_speedup=1.0`, `was_cut=False`, output length == `round(target_dur*sr)`.
4. `fit_to_window` — input 1.10× too long → speeds up to fit, `applied_speedup ≈ input_dur/target_dur` and ≤ 1.25, `was_cut=False`.
5. `fit_to_window` — input 1.5× too long → applies max 1.25× speed-up, then cuts remainder, `was_cut=True`, output length == `round(target_dur*sr)`.
6. `fit_to_window` — input within ±1 ms of target → no speed-up, no cut.
7. `assemble_timeline` — given block waveforms aligned to `start_s`, output length is `round(last_block.end_s * sr)`, gaps between blocks are exact silence, first block starts at sample `round(start_s*sr)`.
8. `assemble_timeline` — handles SRT that starts at `00:00:05` (5-second leading silence).

Use a fixed `sr=24000` in tests; sample rate is a parameter, not a constant.

## Implementation

Public API:

```python
def is_silent(wav: np.ndarray, *, peak_thresh: float = 1e-3, rms_thresh: float = 1e-4) -> bool: ...

def trim_edge_silence(
    wav: np.ndarray, sr: int, *, threshold_db: float = -45.0, min_keep_ms: int = 20
) -> tuple[np.ndarray, int, int]:
    """Returns (trimmed, lead_samples_trimmed, trail_samples_trimmed)."""

def fit_to_window(
    wav: np.ndarray, target_dur_s: float, sr: int, *, max_speedup: float = 1.25
) -> tuple[np.ndarray, float, bool]:
    """Returns (fitted_wav, applied_speedup, was_cut). Output length == round(target_dur_s*sr)."""

def assemble_timeline(
    items: list[tuple[float, np.ndarray]], sr: int, *, total_dur_s: float | None = None
) -> np.ndarray:
    """items = [(start_s, wav), ...]. Pads/positions blocks; returns float32 mono."""
```

Implementation notes:
- `trim_edge_silence` uses a sliding RMS window (~10 ms) and a relative `threshold_db` below the file peak (capped at `peak_thresh` floor) so quiet but valid speech is preserved. `min_keep_ms` prevents accidental over-trimming.
- `fit_to_window` flow:
  1. Compute `input_dur = len(wav)/sr` and `target_len = round(target_dur_s*sr)`.
  2. If `input_dur <= target_dur_s + 1e-3`: pad with zeros to `target_len`, return `(out, 1.0, False)`.
  3. Else compute `needed_speedup = input_dur / target_dur_s`.
  4. If `needed_speedup <= max_speedup`: time-stretch by `needed_speedup`, then trim/pad to exact `target_len`, return `(out, needed_speedup, False)`.
  5. Else: time-stretch by `max_speedup`, hard-cut to `target_len`, return `(out, max_speedup, True)`.
- Time-stretch: prefer `librosa.effects.time_stretch`. If `librosa` is unavailable, fall back to a simple in-repo WSOLA implementation in the same module (gated by `try/except ImportError`). Document the fallback in a one-line comment.
- `assemble_timeline` places each block at sample `round(start_s*sr)`, mixing by direct write (later blocks overwrite gap silence; overlap is impossible since Phase 1 rejects it). Final length = `total_dur_s` if given, else `max(end_of_last_block_sample)`.

## Success Criteria

- [ ] All tests pass with deterministic synthetic signals (sine + zeros).
- [ ] No dependency added beyond what `librosa.effects.time_stretch` needs; if `librosa` is rejected, the WSOLA fallback covers all `fit_to_window` tests.
- [ ] Module is import-safe in environments without `librosa` (lazy import inside the function).
