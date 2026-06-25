"""SRT-to-audio pipeline orchestrator.

UI-agnostic generator that:
1. validates SRT,
2. for each block calls a provided `synthesize_chunk(text) -> np.ndarray`,
3. retries on silent output up to `max_silent_retries`,
4. trims edge silence, fits to the block's [start, end] window,
5. assembles the final timeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generator, Optional

import numpy as np

from .srt_audio_ops import (
    assemble_timeline,
    fit_to_window,
    is_silent,
    trim_edge_silence,
)
from .srt_parser import SrtBlock, SrtValidationError, parse_srt

# Upper bound on total timeline length the SRT pipeline will allocate.
# Prevents user-controlled timestamps (parser accepts up to 99:59:59,999)
# from triggering tens of GB of zero-filled output.
MAX_TOTAL_DURATION_S: float = 2 * 60 * 60  # 2 hours


@dataclass
class ChunkLog:
    index: int
    start_s: float
    end_s: float
    retries: int = 0
    trim_lead_ms: int = 0
    trim_trail_ms: int = 0
    silence_sacrificed_ms: int = 0
    pad_ms: int = 0
    speedup: float = 1.0
    cut_ms: int = 0
    status: str = "ok"  # ok | retried | cut | failed | cancelled
    failure_reason: str = ""


class SrtGenerationError(RuntimeError):
    def __init__(self, message: str, *, block: int):
        super().__init__(message)
        self.block = block


def _fmt_ts(s: float) -> str:
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s - (h * 3600 + m * 60)
    return f"{h:02d}:{m:02d}:{sec:06.3f}"


def format_chunk_log(log: ChunkLog) -> str:
    """One concise line per block, suitable for the UI status area."""
    parts = [
        f"Khối {log.index:02d} ",
        f"[{_fmt_ts(log.start_s)}→{_fmt_ts(log.end_s)}] ",
        f"retries={log.retries} ",
        f"trim={log.trim_lead_ms}/{log.trim_trail_ms}ms ",
    ]
    if log.silence_sacrificed_ms:
        parts.append(f"sil_cut={log.silence_sacrificed_ms}ms ")
    parts += [
        f"pad={log.pad_ms}ms ",
        f"speedup={log.speedup:.2f}x ",
        f"cut={log.cut_ms}ms ",
        log.status,
    ]
    if log.failure_reason:
        parts.append(f" ({log.failure_reason})")
    return "".join(parts)


def synthesize_srt(
    srt_text: str,
    *,
    synthesize_chunk: Callable[[str], np.ndarray],
    sr: int,
    max_speedup: float = 1.25,
    max_silent_retries: int = 2,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Generator[tuple, None, tuple[np.ndarray, list[ChunkLog]]]:
    """Generator. Yields ('status', message) progress events; returns
    `(final_wav, logs)` via StopIteration.value when done.
    Raises SrtValidationError before any synth call if input is invalid.
    Raises SrtGenerationError on persistent silent output, with .block set.
    """
    blocks: list[SrtBlock] = parse_srt(srt_text)
    n = len(blocks)
    if blocks and blocks[-1].end_s > MAX_TOTAL_DURATION_S:
        raise SrtValidationError(
            "duration_exceeded",
            (
                f"Tổng thời lượng SRT vượt giới hạn "
                f"({blocks[-1].end_s:.1f}s > {MAX_TOTAL_DURATION_S:.0f}s)."
            ),
        )
    yield ("status", f"Đã xác thực {n} khối phụ đề.")

    items: list[tuple[float, np.ndarray]] = []
    logs: list[ChunkLog] = []

    for i, block in enumerate(blocks, start=1):
        log = ChunkLog(index=block.index, start_s=block.start_s, end_s=block.end_s)
        target_dur = block.end_s - block.start_s

        wav: Optional[np.ndarray] = None
        for attempt in range(max_silent_retries + 1):
            wav = synthesize_chunk(block.text)
            wav = np.asarray(wav, dtype=np.float32)
            if not is_silent(wav):
                log.retries = attempt
                break
            log.retries = attempt
            if attempt < max_silent_retries:
                yield (
                    "status",
                    f"Khối {block.index}: âm thanh trống, thử lại "
                    f"({attempt + 1}/{max_silent_retries})…",
                )
        else:
            log.status = "failed"
            log.failure_reason = "silent output after retries"
            logs.append(log)
            raise SrtGenerationError(
                f"Khối {block.index}: TTS trả về âm thanh trống sau "
                f"{max_silent_retries} lần thử lại.",
                block=block.index,
            )

        # Edge-silence trim (preserves natural padding).
        trimmed, lead, trail, lead_pad, trail_pad = trim_edge_silence(wav, sr)
        log.trim_lead_ms = int(round(lead / sr * 1000))
        log.trim_trail_ms = int(round(trail / sr * 1000))

        # Fit to window (sacrifices silence padding before speedup/cut).
        fitted, applied_speedup, was_cut, sil_trimmed = fit_to_window(
            trimmed, target_dur_s=target_dur, sr=sr, max_speedup=max_speedup,
            lead_pad_samples=lead_pad, trail_pad_samples=trail_pad,
        )
        log.speedup = float(applied_speedup)
        log.silence_sacrificed_ms = int(round(sil_trimmed / sr * 1000))

        effective_dur = (len(trimmed) - sil_trimmed) / float(sr)
        if effective_dur < target_dur:
            log.pad_ms = int(round((target_dur - effective_dur) * 1000))
        if was_cut:
            stretched_dur = effective_dur / max_speedup
            log.cut_ms = max(0, int(round((stretched_dur - target_dur) * 1000)))

        # Status: cut > retried > ok.
        if was_cut:
            log.status = "cut"
        elif log.retries > 0:
            log.status = "retried"
        else:
            log.status = "ok"

        items.append((block.start_s, fitted))
        logs.append(log)
        yield ("status", f"Khối {i}/{n}: {format_chunk_log(log)}")

        if should_stop is not None and should_stop():
            # Mark the last log as cancelled and stop.
            logs[-1].status = "cancelled"
            yield ("status", f"⏹️ Dừng sau khối {i}/{n}.")
            break

    total_dur = blocks[-1].end_s if not items else max(b.end_s for b in blocks[: len(items)])
    final = assemble_timeline(items, sr, total_dur_s=total_dur)
    return final, logs
