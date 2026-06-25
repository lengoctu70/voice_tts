import numpy as np
import pytest

from vieneu_utils.srt_parser import SrtValidationError
from vieneu_utils.srt_to_audio import (
    MAX_TOTAL_DURATION_S,
    ChunkLog,
    SrtGenerationError,
    format_chunk_log,
    synthesize_srt,
)


SR = 24000


def _tone(dur_s: float, sr: int = SR) -> np.ndarray:
    n = int(round(dur_s * sr))
    t = np.arange(n, dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _silence(dur_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(round(dur_s * sr)), dtype=np.float32)


def _drive(gen):
    """Consume a synthesize_srt generator; return (events, final)."""
    events = []
    final = None
    while True:
        try:
            ev = next(gen)
        except StopIteration as e:
            final = e.value
            break
        events.append(ev)
    return events, final


HAPPY_SRT = (
    "1\n00:00:00,000 --> 00:00:01,000\nA\n\n"
    "2\n00:00:01,500 --> 00:00:02,500\nB\n\n"
    "3\n00:00:03,000 --> 00:00:04,000\nC\n"
)


def test_happy_path():
    durations = {"A": 0.9, "B": 0.95, "C": 0.9}

    def stub(text):
        return _tone(durations[text])

    events, final = _drive(synthesize_srt(HAPPY_SRT, synthesize_chunk=stub, sr=SR))
    assert final is not None
    wav, logs = final
    assert len(wav) == int(round(4.0 * SR))
    assert all(isinstance(l, ChunkLog) for l in logs)
    assert all(l.status == "ok" for l in logs)
    # at least one status event per block
    status_strs = [e[1] for e in events if e[0] == "status"]
    assert sum(1 for s in status_strs if "Khối 1" in s or "1/3" in s) >= 1


def test_padding_branch():
    srt = "1\n00:00:00,000 --> 00:00:02,000\nA\n"

    def stub(text):
        return _tone(0.5)

    _, final = _drive(synthesize_srt(srt, synthesize_chunk=stub, sr=SR))
    wav, logs = final
    assert logs[0].pad_ms > 1000
    assert logs[0].speedup == pytest.approx(1.0)
    assert logs[0].cut_ms == 0


def test_speedup_within_cap():
    srt = "1\n00:00:00,000 --> 00:00:01,000\nA\n"

    def stub(text):
        return _tone(1.10)

    _, final = _drive(synthesize_srt(srt, synthesize_chunk=stub, sr=SR))
    _, logs = final
    assert logs[0].speedup == pytest.approx(1.10, abs=0.05)
    assert logs[0].cut_ms == 0
    assert logs[0].status == "ok"


def test_cut_beyond_cap():
    srt = "1\n00:00:00,000 --> 00:00:01,000\nA\n"

    def stub(text):
        return _tone(2.0)

    _, final = _drive(synthesize_srt(srt, synthesize_chunk=stub, sr=SR))
    _, logs = final
    assert logs[0].speedup == pytest.approx(1.25, abs=1e-6)
    assert logs[0].cut_ms > 0
    assert logs[0].status == "cut"


def test_silent_retry_success():
    srt = "1\n00:00:00,000 --> 00:00:01,000\nA\n"
    calls = {"n": 0}

    def stub(text):
        calls["n"] += 1
        if calls["n"] <= 2:
            return _silence(1.0)
        return _tone(1.0)

    _, final = _drive(synthesize_srt(srt, synthesize_chunk=stub, sr=SR))
    _, logs = final
    assert calls["n"] == 3
    assert logs[0].retries == 2
    assert logs[0].status == "retried"


def test_silent_retry_fail():
    srt = "1\n00:00:00,000 --> 00:00:01,000\nA\n"

    def stub(text):
        return _silence(1.0)

    with pytest.raises(SrtGenerationError) as ei:
        _drive(synthesize_srt(srt, synthesize_chunk=stub, sr=SR))
    assert ei.value.block == 1


def test_invalid_srt_skips_synth():
    calls = {"n": 0}

    def stub(text):
        calls["n"] += 1
        return _tone(1.0)

    with pytest.raises(SrtValidationError):
        _drive(synthesize_srt("not srt", synthesize_chunk=stub, sr=SR))
    assert calls["n"] == 0


def test_stop_signal_after_block_1():
    stop_after = {"after": False}

    def should_stop():
        return stop_after["after"]

    def stub(text):
        return _tone(0.9)

    gen = synthesize_srt(
        HAPPY_SRT, synthesize_chunk=stub, sr=SR, should_stop=should_stop
    )
    # consume until first chunk done, then flip stop
    saw_first = False
    events = []
    final = None
    while True:
        try:
            ev = next(gen)
        except StopIteration as e:
            final = e.value
            break
        events.append(ev)
        if ev[0] == "status" and "Khối 1" in ev[1] and not saw_first:
            saw_first = True
            stop_after["after"] = True

    assert final is not None
    _, logs = final
    statuses = [l.status for l in logs]
    assert "cancelled" in statuses


def test_rejects_srt_exceeding_max_total_duration():
    # End timestamp just past the cap; SR-side allocation never reached.
    end_h = int(MAX_TOTAL_DURATION_S // 3600) + 1
    srt = (
        "1\n"
        f"00:00:00,000 --> {end_h:02d}:00:00,000\n"
        "vượt giới hạn\n"
    )

    def synth(_text):
        raise AssertionError("synth must not be called when SRT exceeds cap")

    gen = synthesize_srt(srt, synthesize_chunk=synth, sr=SR)
    with pytest.raises(SrtValidationError):
        next(gen)


def test_format_chunk_log_concise():
    log = ChunkLog(
        index=3,
        start_s=5.2,
        end_s=7.8,
        retries=0,
        trim_lead_ms=12,
        trim_trail_ms=40,
        speedup=1.10,
        status="ok",
    )
    s = format_chunk_log(log)
    assert "Khối 03" in s or "Khối 3" in s
    assert "1.10" in s
