import numpy as np
import pytest

from vieneu_utils.srt_audio_ops import (
    assemble_timeline,
    fit_to_window,
    is_silent,
    trim_edge_silence,
)


SR = 24000


def _tone(dur_s: float, freq: float = 440.0, amp: float = 0.5, sr: int = SR) -> np.ndarray:
    n = int(round(dur_s * sr))
    t = np.arange(n, dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _silence(dur_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(round(dur_s * sr)), dtype=np.float32)


def test_is_silent_all_zero():
    assert is_silent(np.zeros(1024, dtype=np.float32)) is True


def test_is_silent_tone_false():
    assert is_silent(_tone(0.1)) is False


def test_is_silent_below_threshold():
    buf = (np.ones(2048, dtype=np.float32) * 5e-5)
    assert is_silent(buf) is True


def test_trim_edge_silence_keeps_inner_gap():
    signal = np.concatenate(
        [
            _silence(0.2),
            _tone(0.1),
            _silence(0.05),
            _tone(0.1),
            _silence(0.2),
        ]
    )
    trimmed, lead, trail, lead_pad, trail_pad = trim_edge_silence(
        signal, SR, preserve_lead_ms=0, preserve_trail_ms=0,
    )
    assert lead > int(0.15 * SR)
    assert trail > int(0.15 * SR)
    assert lead_pad == 0
    assert trail_pad == 0
    # inner gap preserved -> trimmed length is roughly 0.25s (two tones + inner gap)
    assert abs(len(trimmed) - int(0.25 * SR)) < int(0.05 * SR)


def test_trim_edge_silence_preserves_padding():
    signal = np.concatenate(
        [
            _silence(0.2),
            _tone(0.1),
            _silence(0.2),
        ]
    )
    trimmed, lead, trail, lead_pad, trail_pad = trim_edge_silence(
        signal, SR, preserve_lead_ms=50, preserve_trail_ms=150,
    )
    expected_lead_pad = int(round(0.050 * SR))
    expected_trail_pad = int(round(0.150 * SR))
    assert lead_pad == expected_lead_pad
    assert trail_pad == expected_trail_pad
    assert lead > 0
    assert trail > 0


def test_fit_to_window_pads_short():
    inp = _tone(0.5)
    out, speedup, was_cut, sil_trimmed = fit_to_window(inp, target_dur_s=2.0, sr=SR)
    assert len(out) == int(round(2.0 * SR))
    assert speedup == pytest.approx(1.0)
    assert was_cut is False
    assert sil_trimmed == 0
    # Tail must be silence after the tone.
    assert np.all(np.abs(out[int(0.55 * SR):]) < 1e-6)


def test_fit_to_window_speedup_within_cap():
    inp = _tone(1.10)
    out, speedup, was_cut, _ = fit_to_window(inp, target_dur_s=1.0, sr=SR)
    assert len(out) == int(round(1.0 * SR))
    assert speedup == pytest.approx(1.10, abs=0.02)
    assert speedup <= 1.25 + 1e-6
    assert was_cut is False


def test_fit_to_window_cut_beyond_cap():
    inp = _tone(1.5)
    out, speedup, was_cut, _ = fit_to_window(inp, target_dur_s=1.0, sr=SR)
    assert len(out) == int(round(1.0 * SR))
    assert speedup == pytest.approx(1.25, abs=1e-6)
    assert was_cut is True


def test_fit_to_window_borderline_no_op():
    inp = _tone(1.0)
    out, speedup, was_cut, _ = fit_to_window(inp, target_dur_s=1.0, sr=SR)
    assert len(out) == int(round(1.0 * SR))
    assert speedup == pytest.approx(1.0, abs=1e-6)
    assert was_cut is False


def test_fit_to_window_sacrifices_silence_before_speedup():
    """When audio has padding silence and is slightly too long, silence is
    trimmed first and no speedup is applied."""
    tone = _tone(1.0)
    pad_trail = _silence(0.1)
    inp = np.concatenate([tone, pad_trail])
    # 1.1s audio, 1.0s target, 0.1s trail padding available.
    out, speedup, was_cut, sil_trimmed = fit_to_window(
        inp, target_dur_s=1.0, sr=SR,
        trail_pad_samples=len(pad_trail),
    )
    assert len(out) == int(round(1.0 * SR))
    assert speedup == pytest.approx(1.0)
    assert was_cut is False
    assert sil_trimmed == len(pad_trail)


def test_fit_to_window_partial_silence_then_speedup():
    """When sacrificing all silence still leaves audio too long, it trims
    silence first then applies minimal speedup."""
    tone = _tone(1.2)
    pad_trail = _silence(0.1)
    inp = np.concatenate([tone, pad_trail])
    # 1.3s audio, 1.0s target, only 0.1s trail padding.
    out, speedup, was_cut, sil_trimmed = fit_to_window(
        inp, target_dur_s=1.0, sr=SR,
        trail_pad_samples=len(pad_trail),
    )
    assert len(out) == int(round(1.0 * SR))
    assert sil_trimmed == len(pad_trail)
    # After trimming 0.1s silence, 1.2s remains -> speedup ~1.2.
    assert speedup == pytest.approx(1.2, abs=0.02)
    assert was_cut is False


def test_assemble_timeline_basic():
    b1 = _tone(0.5)
    b2 = _tone(0.5)
    items = [(0.0, b1), (1.0, b2)]
    out = assemble_timeline(items, SR, total_dur_s=1.5)
    assert len(out) == int(round(1.5 * SR))
    # Block 1 starts at sample 0.
    assert np.any(np.abs(out[:100]) > 0.01)
    # Gap between samples 0.5s..1.0s is silence.
    gap = out[int(0.5 * SR):int(1.0 * SR)]
    assert np.allclose(gap, 0.0)
    # Block 2 starts at 1.0s
    assert np.any(np.abs(out[int(1.0 * SR):int(1.0 * SR) + 100]) > 0.01)


def test_assemble_timeline_leading_silence():
    block = _tone(0.5)
    out = assemble_timeline([(5.0, block)], SR, total_dur_s=5.5)
    assert len(out) == int(round(5.5 * SR))
    # First 5 seconds silence
    assert np.allclose(out[: int(5.0 * SR)], 0.0)
    # Audio at 5s
    assert np.any(np.abs(out[int(5.0 * SR): int(5.0 * SR) + 100]) > 0.01)
