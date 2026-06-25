"""Audio post-processing utilities for the SRT-to-audio pipeline.

Pure-NumPy operations: silence detection, edge trim, fit-to-window
(pad / time-stretch / hard-cut), and timeline assembly.

Time-stretch prefers `librosa.effects.time_stretch` (high quality); if
librosa is unavailable, an in-module SOLA fallback is used.
"""
from __future__ import annotations

import numpy as np


def is_silent(
    wav: np.ndarray,
    *,
    peak_thresh: float = 1e-3,
    rms_thresh: float = 1e-4,
) -> bool:
    if wav.size == 0:
        return True
    peak = float(np.max(np.abs(wav)))
    if peak >= peak_thresh:
        return False
    rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))
    return rms < rms_thresh


def _frame_rms(wav: np.ndarray, frame_len: int) -> tuple[np.ndarray, int]:
    """Compute non-overlapping RMS per frame. Returns (rms_per_frame, frame_len)."""
    n = len(wav)
    if frame_len <= 0 or n == 0:
        return np.zeros(0, dtype=np.float32), frame_len
    n_frames = max(1, n // frame_len)
    trimmed = wav[: n_frames * frame_len].astype(np.float64, copy=False)
    framed = trimmed.reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(framed * framed, axis=1)).astype(np.float32)
    return rms, frame_len


def trim_edge_silence(
    wav: np.ndarray,
    sr: int,
    *,
    threshold_db: float = -45.0,
    min_keep_ms: int = 20,
) -> tuple[np.ndarray, int, int]:
    """Trim silence at the start and end. Preserves internal silence.

    Returns (trimmed_wav, lead_samples_trimmed, trail_samples_trimmed).
    """
    if wav.size == 0:
        return wav.astype(np.float32, copy=False), 0, 0

    frame_len = max(1, int(0.010 * sr))  # 10 ms
    rms, _ = _frame_rms(wav, frame_len)
    if rms.size == 0:
        return wav.astype(np.float32, copy=False), 0, 0

    peak_rms = float(rms.max())
    thresh = max(peak_rms * (10.0 ** (threshold_db / 20.0)), 1e-4)

    above = np.where(rms > thresh)[0]
    if above.size == 0:
        # All silence; keep the whole thing (caller decides what to do).
        return wav.astype(np.float32, copy=False), 0, 0

    first_frame = int(above[0])
    last_frame = int(above[-1])
    keep_margin = max(0, int(round(min_keep_ms / 1000.0 * sr)))

    lead = max(0, first_frame * frame_len - keep_margin)
    trail_end = min(len(wav), (last_frame + 1) * frame_len + keep_margin)

    trimmed = wav[lead:trail_end].astype(np.float32, copy=False)
    return trimmed, lead, len(wav) - trail_end


def _time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch by `rate` (>1 = faster/shorter). Returns float32 mono.

    Uses librosa if available; otherwise falls back to a simple SOLA.
    """
    if abs(rate - 1.0) < 1e-6 or y.size == 0:
        return y.astype(np.float32, copy=False)

    try:
        import librosa  # type: ignore

        out = librosa.effects.time_stretch(
            y.astype(np.float32, copy=False), rate=float(rate)
        )
        return out.astype(np.float32, copy=False)
    except Exception:
        pass

    # SOLA fallback (pure NumPy). Good enough for short speech and the
    # ≤1.25× cap; phase artifacts are tolerable below 1.25× for prosody.
    y32 = y.astype(np.float32, copy=False)
    frame = 1024
    hop_s = frame // 4
    hop_a = max(1, int(round(hop_s * float(rate))))
    win = np.hanning(frame).astype(np.float32)

    if len(y32) <= frame:
        # Too short to frame; resample by linear interpolation as a last resort.
        new_n = max(1, int(round(len(y32) / float(rate))))
        x_old = np.linspace(0.0, 1.0, num=len(y32), endpoint=False, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, num=new_n, endpoint=False, dtype=np.float32)
        return np.interp(x_new, x_old, y32).astype(np.float32)

    n_frames = 1 + (len(y32) - frame) // hop_a
    out_len = hop_s * (n_frames - 1) + frame
    out = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    for i in range(n_frames):
        a = i * hop_a
        s = i * hop_s
        seg = y32[a : a + frame]
        if seg.size < frame:
            seg = np.pad(seg, (0, frame - seg.size))
        out[s : s + frame] += seg * win
        norm[s : s + frame] += win

    out /= np.maximum(norm, 1e-6)
    return out.astype(np.float32, copy=False)


def _pad_or_trim_exact(wav: np.ndarray, target_len: int) -> np.ndarray:
    if len(wav) == target_len:
        return wav.astype(np.float32, copy=False)
    if len(wav) > target_len:
        return wav[:target_len].astype(np.float32, copy=False)
    out = np.zeros(target_len, dtype=np.float32)
    out[: len(wav)] = wav
    return out


def fit_to_window(
    wav: np.ndarray,
    target_dur_s: float,
    sr: int,
    *,
    max_speedup: float = 1.25,
) -> tuple[np.ndarray, float, bool]:
    """Fit a waveform to an exact target duration.

    Rules:
      1. shorter or within +1 ms → pad with trailing silence.
      2. longer but needed_speedup ≤ max_speedup → time-stretch to fit.
      3. longer beyond cap → stretch at max_speedup then hard-cut.

    Returns (fitted_wav, applied_speedup, was_cut).
    """
    target_len = int(round(target_dur_s * sr))
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32), 1.0, False

    wav = wav.astype(np.float32, copy=False)
    input_dur = len(wav) / float(sr)
    tol = 1e-3  # 1 ms

    if input_dur <= target_dur_s + tol:
        return _pad_or_trim_exact(wav, target_len), 1.0, False

    needed_speedup = input_dur / target_dur_s
    if needed_speedup <= max_speedup + 1e-6:
        stretched = _time_stretch(wav, needed_speedup)
        return _pad_or_trim_exact(stretched, target_len), float(needed_speedup), False

    stretched = _time_stretch(wav, max_speedup)
    return _pad_or_trim_exact(stretched, target_len), float(max_speedup), True


def assemble_timeline(
    items: list[tuple[float, np.ndarray]],
    sr: int,
    *,
    total_dur_s: float | None = None,
) -> np.ndarray:
    """Place each (start_s, wav) at sample round(start_s*sr) on a zero canvas.

    Output dtype is float32 mono. If `total_dur_s` is omitted, length is the
    sample index of the last item's end.
    """
    if total_dur_s is not None:
        out_len = int(round(total_dur_s * sr))
    else:
        out_len = 0
        for start_s, wav in items:
            out_len = max(out_len, int(round(start_s * sr)) + len(wav))

    out = np.zeros(out_len, dtype=np.float32)
    for start_s, wav in items:
        start_idx = int(round(start_s * sr))
        end_idx = min(out_len, start_idx + len(wav))
        usable = max(0, end_idx - start_idx)
        if usable > 0:
            out[start_idx:end_idx] = wav[:usable].astype(np.float32, copy=False)
    return out
