---
phase: 1
title: "SRT parser & validator"
status: completed
effort: "S"
---

# Phase 1: SRT parser & validator

## Overview

Pure-Python parser and validator for SRT subtitle text. No TTS dependency. Returns a list of `SrtBlock` (`index`, `start_s`, `end_s`, `text`). Validation rejects malformed timestamps, missing text, wrong block order, overlapping ranges, zero/negative durations, and empty input — each with a specific message identifying the offending block.

## Files

- Create `src/vieneu_utils/srt_parser.py`.
- Create `tests/test_srt_parser.py`.

## TDD — Tests first

Write `tests/test_srt_parser.py` covering:

1. **Happy path** — 3-block SRT parses with correct indices, float seconds, exact text (including multi-line subtitle text joined with `\n`).
2. **Empty input** — empty string and whitespace-only raise `SrtValidationError("empty")`.
3. **Malformed timestamp** — `00:00:01 --> 00:00:03` (missing milliseconds) raises with the line number.
4. **Missing text** — block with timing line but no following text line raises identifying the block index.
5. **Blank text** — block whose text line is empty/whitespace raises identifying the block index.
6. **Out-of-order indices** — index 3 before 2 raises identifying both.
7. **Overlap** — block N+1 `start_s` < block N `end_s` raises identifying both indices.
8. **Zero / negative duration** — `start_s >= end_s` raises identifying the block.
9. **BOM and CRLF tolerance** — parser handles UTF-8 BOM and `\r\n` line endings.
10. **Long gap is allowed** — large gap between blocks parses fine (gap handled later in pipeline).

Each test asserts `SrtValidationError.code` ∈ `{"empty","malformed_timestamp","missing_text","blank_text","bad_order","overlap","bad_duration"}` and a human-readable `.message`.

## Implementation

Public API:

```python
@dataclass(frozen=True)
class SrtBlock:
    index: int
    start_s: float
    end_s: float
    text: str

class SrtValidationError(ValueError):
    def __init__(self, code: str, message: str, *, block: int | None = None): ...

def parse_srt(raw: str) -> list[SrtBlock]:
    """Parse + validate. Raises SrtValidationError on any failure."""
```

Parsing steps:
1. Strip UTF-8 BOM, normalize line endings to `\n`, split on blank-line boundaries.
2. For each block: first non-empty line = index; next line must match `HH:MM:SS,mmm --> HH:MM:SS,mmm`; remaining lines joined with `\n` form the text.
3. Run validators in sequence and raise on first failure: empty → malformed_timestamp → missing_text → blank_text → bad_duration → bad_order → overlap.

Time parsing helper:

```python
_RE_TS = re.compile(r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})$")
```

Accept both `,` and `.` as the millisecond separator (lenient).

## Success Criteria

- [ ] All test cases in `tests/test_srt_parser.py` pass.
- [ ] No new third-party dependency.
- [ ] `parse_srt` returns blocks sorted by input order with floats accurate to ≥1 ms.
- [ ] Each `SrtValidationError` carries a `code` from the fixed set and references the offending block index when applicable.
