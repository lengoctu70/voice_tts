---
phase: 3
title: "SRT-to-audio pipeline orchestrator"
status: completed
effort: "M"
---

# Phase 3: SRT-to-audio pipeline orchestrator

## Overview

Generator function that drives the full pipeline: validate SRT, for each block synthesize via the supplied TTS callable, trim/fit, retry on silent output (≤2 retries), assemble timeline, return final `(wav, sr, chunk_log)`. UI-agnostic — accepts a `synthesize_chunk` callable so it can be tested without loading any model.

## Files

- Create `src/vieneu_utils/srt_to_audio.py`.
- Create `tests/test_srt_to_audio.py`.

## TDD — Tests first

`tests/test_srt_to_audio.py` uses a stub `synthesize_chunk(text) -> np.ndarray`:

1. **Happy path** — 3 blocks; stub returns short tones; final wav length matches `last.end_s*sr`, each block placed at `start_s`, all `ChunkLog.status == "ok"`.
2. **Padding** — stub returns 0.5 s tone for a 2.0 s block; log shows `pad_ms ≈ 1500`, `speedup == 1.0`, `cut_ms == 0`.
3. **Speed-up under 1.25×** — stub returns 1.10 s tone for a 1.0 s block; log shows `speedup ≈ 1.10`, `cut_ms == 0`.
4. **Cut after 1.25× cap** — stub returns 2.0 s tone for a 1.0 s block; log shows `speedup == 1.25`, `cut_ms > 0`, `status == "cut"`.
5. **Silent retry success** — stub returns silence on calls 1–2, audible tone on call 3; log shows `retries == 2`, `status == "retried"`, job succeeds.
6. **Silent retry fail** — stub always returns silence; job raises `SrtGenerationError` with the failing block index, no partial WAV returned.
7. **Invalid SRT** — invalid input raises `SrtValidationError` before any call to `synthesize_chunk` (stub asserts call count == 0).
8. **Progress events** — generator yields status strings of the form `"chunk i/N: …"` and at least one per block.
9. **Stop signal** — passing a `should_stop()` callable that returns True after block 1 stops generation cleanly with a `cancelled` status.

## Implementation

Public API:

```python
@dataclass
class ChunkLog:
    index: int
    start_s: float
    end_s: float
    retries: int = 0
    trim_lead_ms: int = 0
    trim_trail_ms: int = 0
    pad_ms: int = 0
    speedup: float = 1.0
    cut_ms: int = 0
    status: str = "ok"        # ok | retried | cut | failed | cancelled
    failure_reason: str = ""

class SrtGenerationError(RuntimeError):
    def __init__(self, message: str, *, block: int):
        super().__init__(message); self.block = block

def synthesize_srt(
    srt_text: str,
    *,
    synthesize_chunk,           # Callable[[str], np.ndarray]
    sr: int,
    max_speedup: float = 1.25,
    max_silent_retries: int = 2,
    should_stop=None,           # Callable[[], bool] | None
):
    """Generator. Yields ('status', str) progress events.
    Final yield: ('done', wav: np.ndarray, logs: list[ChunkLog])."""
```

Flow per block:
1. `wav = synthesize_chunk(text)`.
2. If `is_silent(wav)` and `retries < max_silent_retries`: increment retries, repeat step 1.
3. If still silent: raise `SrtGenerationError(f"silent output", block=idx)` with `ChunkLog.status="failed"`.
4. `wav, lead, trail = trim_edge_silence(wav, sr)`; record `trim_lead_ms`, `trim_trail_ms`.
5. `wav, speedup, was_cut = fit_to_window(wav, end_s - start_s, sr, max_speedup=max_speedup)`.
6. Record `pad_ms` if `input_dur < target_dur`; record `cut_ms` if `was_cut`; pick `status` accordingly (`cut` > `retried` > `ok`).
7. Append `(start_s, wav)` to assembly list.
8. Between blocks, check `should_stop()`; if True, set last log `status="cancelled"` and break.

After loop: `final = assemble_timeline(items, sr, total_dur_s=blocks[-1].end_s)`.

Diagnostics formatter helper (used by UI in Phase 4):

```python
def format_chunk_log(log: ChunkLog) -> str:
    """One concise line: 'Block 03 [00:00:05.200→00:00:07.800] retries=0 trim=12/40ms speedup=1.10x cut=0ms ok'"""
```

## Success Criteria

- [ ] All tests pass without loading any TTS model (pure stub).
- [ ] On silent-retry failure, no partial final WAV is yielded.
- [ ] Generator never calls `synthesize_chunk` if SRT validation fails.
- [ ] `ChunkLog.status` values are limited to the documented set.
