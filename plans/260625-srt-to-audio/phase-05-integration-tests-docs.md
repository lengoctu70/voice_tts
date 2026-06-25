---
phase: 5
title: "Integration tests & docs"
status: completed
effort: "S"
---

# Phase 5: Integration tests & docs

## Overview

End-to-end test that drives the full Phase 3 pipeline with a deterministic stub TTS to confirm every acceptance-criterion branch, plus minimal documentation updates so users discover the feature.

## Files

- Create `tests/test_srt_to_audio_integration.py`.
- Create fixture `tests/fixtures/sample_srt_basic.srt` (3 blocks: short, medium-needs-speedup, long-needs-cut, with a 1-second gap somewhere).
- Update `docs/project-overview-pdr.md` and `docs/codebase-summary.md`: short paragraph + module references.
- Update `README.md` and `README.vi.md`: one-line feature bullet and a short "SRT → Audio" subsection under the Web UI section pointing at the new tab.

## TDD — Integration test

`tests/test_srt_to_audio_integration.py`:

1. Loads `sample_srt_basic.srt`.
2. Defines a stub `synthesize_chunk` whose return length is computed from the block text length so each acceptance-criterion branch (pad, speedup, cut, retry-success, retry-fail) is exercised in distinct sub-tests.
3. Calls `vieneu_utils.srt_to_audio.synthesize_srt(...)` end-to-end, collecting events to completion.
4. Asserts:
   - Final wav length == `round(last_block.end_s * sr)` (±1 sample).
   - Block N starts at sample `round(start_s * sr)` (verified by locating the tone's first non-zero sample in a window around the expected offset).
   - `ChunkLog` entries match expected `status`, `speedup`, `cut_ms`, `pad_ms`, `retries`.
   - Silent-fail path raises `SrtGenerationError` carrying the right block index and no WAV is produced.
5. Validates invalid-SRT path raises `SrtValidationError` with a specific `code` for: empty, overlap, bad_order, bad_duration, blank_text, malformed_timestamp.

## Docs

- `docs/project-overview-pdr.md`: add 1 paragraph describing the SRT-to-Audio feature, single-speaker scope, and the 1.25× cap.
- `docs/codebase-summary.md`: add bullets for the three new modules and one new tab in `apps/gradio_main.py`.
- `README.md` / `README.vi.md`: 1-line bullet under Key Features ("Đồng bộ phụ đề: dán SRT, tạo một file WAV khớp timeline") and a 3–4-line subsection demonstrating the tab.

## Success Criteria

- [ ] Integration test passes locally (`pytest tests/test_srt_to_audio_integration.py -q`).
- [ ] `pytest tests/` reports all SRT-feature tests green and no regressions in pre-existing tests.
- [ ] Docs updated; the new tab is mentioned in the README's Web UI walkthrough.
- [ ] No new top-level dependency added; if `librosa` was needed in Phase 2, it is declared in `pyproject.toml` under the existing dependency group used by the Web UI (`uv sync` still works).
