from pathlib import Path

import numpy as np
import pytest

from vieneu_utils.srt_parser import SrtValidationError
from vieneu_utils.srt_to_audio import (
    ChunkLog,
    SrtGenerationError,
    synthesize_srt,
)


SR = 24000
FIXTURE = Path(__file__).parent / "fixtures" / "sample_srt_basic.srt"


def _tone(dur_s: float, sr: int = SR) -> np.ndarray:
    n = int(round(dur_s * sr))
    t = np.arange(n, dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _silence(dur_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(round(dur_s * sr)), dtype=np.float32)


def _drive(gen):
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value


def test_fixture_loads_and_parses():
    raw = FIXTURE.read_text(encoding="utf-8")
    assert "00:00:00,000" in raw


def test_end_to_end_pad_speedup_cut():
    raw = FIXTURE.read_text(encoding="utf-8")
    # Block 1 [0..1s]: pad. Block 2 [2..3s]: speedup. Block 3 [4..5s]: cut.
    durations = [0.5, 1.10, 2.0]

    def stub(text):
        # Match by call order — synthesize_srt iterates blocks sequentially.
        stub.idx += 1
        return _tone(durations[stub.idx - 1])

    stub.idx = 0

    final = _drive(synthesize_srt(raw, synthesize_chunk=stub, sr=SR))
    wav, logs = final
    assert len(wav) == int(round(5.0 * SR))
    assert logs[0].pad_ms > 0 and logs[0].status == "ok"
    assert logs[1].speedup == pytest.approx(1.10, abs=0.05)
    assert logs[1].cut_ms == 0
    assert logs[2].status == "cut" and logs[2].cut_ms > 0

    # Block 1 starts at sample 0; block 2 at 2.0s; block 3 at 4.0s.
    for i, start_s in enumerate([0.0, 2.0, 4.0]):
        win_start = int(round(start_s * SR))
        win_end = win_start + 200
        assert np.any(np.abs(wav[win_start:win_end]) > 0.01), f"silent at block {i + 1} start"

    # Gaps between blocks are silent (sample range strictly inside the gap).
    gap = wav[int(1.05 * SR): int(1.95 * SR)]
    assert np.allclose(gap, 0.0)


def test_silent_retry_then_fail_raises_with_block():
    raw = FIXTURE.read_text(encoding="utf-8")

    def stub(text):
        return _silence(1.0)

    with pytest.raises(SrtGenerationError) as ei:
        _drive(synthesize_srt(raw, synthesize_chunk=stub, sr=SR))
    assert ei.value.block == 1


@pytest.mark.parametrize(
    "raw, code",
    [
        ("", "empty"),
        ("1\n00:00:01 --> 00:00:02\nHi.\n", "malformed_timestamp"),
        ("1\n00:00:02,000 --> 00:00:02,000\nHi.\n", "bad_duration"),
        ("1\n00:00:01,000 --> 00:00:02,000\n   \n", "blank_text"),
        (
            "1\n00:00:01,000 --> 00:00:03,000\nA.\n\n"
            "2\n00:00:02,500 --> 00:00:04,000\nB.\n",
            "overlap",
        ),
        (
            "1\n00:00:01,000 --> 00:00:02,000\nA.\n\n"
            "3\n00:00:02,500 --> 00:00:03,000\nC.\n\n"
            "2\n00:00:03,500 --> 00:00:04,000\nB.\n",
            "bad_order",
        ),
    ],
)
def test_invalid_srt_paths(raw, code):
    def stub(text):
        return _tone(1.0)

    with pytest.raises(SrtValidationError) as ei:
        _drive(synthesize_srt(raw, synthesize_chunk=stub, sr=SR))
    assert ei.value.code == code


def test_chunklog_dataclass_shape():
    log = ChunkLog(index=1, start_s=0.0, end_s=1.0)
    assert log.retries == 0
    assert log.status == "ok"
