"""SRT subtitle parser and validator.

Parses SRT text into a list of SrtBlock and validates structure.
No third-party dependencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_VALID_CODES = frozenset(
    {
        "empty",
        "malformed_timestamp",
        "missing_text",
        "blank_text",
        "bad_order",
        "overlap",
        "bad_duration",
        "duration_exceeded",
    }
)

_RE_TS = re.compile(
    r"^(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*$"
)


@dataclass(frozen=True)
class SrtBlock:
    index: int
    start_s: float
    end_s: float
    text: str


class SrtValidationError(ValueError):
    def __init__(self, code: str, message: str, *, block: int | None = None):
        if code not in _VALID_CODES:
            raise AssertionError(f"unknown SrtValidationError code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.block = block


def _ts_to_seconds(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(raw: str) -> list[SrtBlock]:
    """Parse + validate SRT. Raises SrtValidationError on any failure."""
    if raw is None or not raw.strip():
        raise SrtValidationError("empty", "Nội dung SRT trống.")

    # Strip UTF-8 BOM and normalize line endings.
    text = raw.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")

    # Split on blank-line boundaries (one or more empty lines).
    chunks = re.split(r"\n\s*\n", text.strip("\n"))
    blocks: list[SrtBlock] = []

    for raw_chunk in chunks:
        lines = [ln for ln in raw_chunk.split("\n")]
        # Drop leading empty lines inside a chunk (rare but be lenient).
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            continue

        # First non-empty line: index.
        idx_line = lines[0].strip()
        try:
            index = int(idx_line)
        except ValueError:
            raise SrtValidationError(
                "malformed_timestamp",
                f"Chỉ số khối không hợp lệ: {idx_line!r}",
            )

        if len(lines) < 2:
            raise SrtValidationError(
                "malformed_timestamp",
                f"Khối {index}: thiếu dòng thời gian.",
                block=index,
            )

        ts_match = _RE_TS.match(lines[1].strip())
        if not ts_match:
            raise SrtValidationError(
                "malformed_timestamp",
                f"Khối {index}: dòng thời gian không đúng định dạng "
                f"HH:MM:SS,mmm --> HH:MM:SS,mmm: {lines[1]!r}",
                block=index,
            )

        start_s = _ts_to_seconds(*ts_match.group(1, 2, 3, 4))
        end_s = _ts_to_seconds(*ts_match.group(5, 6, 7, 8))

        text_lines = lines[2:]
        if not text_lines:
            raise SrtValidationError(
                "missing_text",
                f"Khối {index}: thiếu nội dung phụ đề.",
                block=index,
            )
        joined = "\n".join(text_lines)
        if not joined.strip():
            raise SrtValidationError(
                "blank_text",
                f"Khối {index}: nội dung phụ đề rỗng.",
                block=index,
            )

        if end_s <= start_s:
            raise SrtValidationError(
                "bad_duration",
                f"Khối {index}: thời lượng <= 0 "
                f"({start_s:.3f}s → {end_s:.3f}s).",
                block=index,
            )

        blocks.append(SrtBlock(index=index, start_s=start_s, end_s=end_s, text=joined))

    if not blocks:
        raise SrtValidationError("empty", "Không tìm thấy khối phụ đề nào.")

    # Order + overlap checks.
    for i in range(1, len(blocks)):
        prev, cur = blocks[i - 1], blocks[i]
        if cur.index <= prev.index:
            raise SrtValidationError(
                "bad_order",
                f"Chỉ số khối không tăng dần: {prev.index} → {cur.index}.",
                block=cur.index,
            )
        if cur.start_s < prev.end_s:
            raise SrtValidationError(
                "overlap",
                f"Khối {cur.index} bắt đầu ({cur.start_s:.3f}s) "
                f"trước khi khối {prev.index} kết thúc ({prev.end_s:.3f}s).",
                block=cur.index,
            )

    return blocks
