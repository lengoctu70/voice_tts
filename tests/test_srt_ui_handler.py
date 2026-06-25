import threading

import numpy as np
import soundfile as sf

import pytest

from apps.srt_ui_handler import synthesize_srt_handler


SR = 24000


def _tone(dur_s: float, sr: int = SR) -> np.ndarray:
    n = int(round(dur_s * sr))
    t = np.arange(n, dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _silence(dur_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(round(dur_s * sr)), dtype=np.float32)


class StubTTS:
    """Minimal stand-in for the Vieneu engine used by the SRT handler."""

    def __init__(self, sample_rate: int = SR, infer_fn=None):
        self.sample_rate = sample_rate
        self.call_count = 0
        self._infer_fn = infer_fn or (lambda text, **kw: _tone(0.9))

    def get_preset_voice(self, v_id):
        return {"codes": np.zeros(8, dtype=np.float32), "text": "ref", "reserved_id": None}

    def encode_reference(self, audio):
        return np.zeros(8, dtype=np.float32)

    def infer(self, text, **kw):
        self.call_count += 1
        return self._infer_fn(text, **kw)


HAPPY_SRT = (
    "1\n00:00:00,000 --> 00:00:01,000\nA\n\n"
    "2\n00:00:01,500 --> 00:00:02,500\nB\n"
)


def _drive(gen):
    return list(gen)


def test_valid_srt_writes_wav():
    tts = StubTTS()
    stop = threading.Event()
    outs = _drive(
        synthesize_srt_handler(
            HAPPY_SRT,
            voice_choice="DefaultVoice",
            custom_audio=None,
            custom_text="",
            mode_tab="preset_mode",
            temperature=0.7,
            max_chars_chunk=256,
            tts=tts,
            current_backbone="VieNeu-TTS-v3-Turbo (Thử nghiệm)",
            model_loaded=True,
            stop_event=stop,
        )
    )
    assert any(isinstance(o[1], str) for o in outs)
    last = outs[-1]
    assert last[0] is not None and last[0].endswith(".wav")
    # WAV is readable
    data, sr_read = sf.read(last[0])
    assert sr_read == SR
    assert len(data) == int(round(2.5 * SR))


def test_invalid_srt_skips_infer():
    tts = StubTTS()
    stop = threading.Event()
    outs = _drive(
        synthesize_srt_handler(
            "not a valid srt",
            voice_choice="DefaultVoice",
            custom_audio=None,
            custom_text="",
            mode_tab="preset_mode",
            temperature=0.7,
            max_chars_chunk=256,
            tts=tts,
            current_backbone="VieNeu-TTS-v3-Turbo (Thử nghiệm)",
            model_loaded=True,
            stop_event=stop,
        )
    )
    assert tts.call_count == 0
    # Final yield has None filepath and an error message.
    assert outs[-1][0] is None
    assert "❌" in outs[-1][1]


def test_silent_failure_surfaces_block_index():
    tts = StubTTS(infer_fn=lambda text, **kw: _silence(1.0))
    stop = threading.Event()
    outs = _drive(
        synthesize_srt_handler(
            HAPPY_SRT,
            voice_choice="DefaultVoice",
            custom_audio=None,
            custom_text="",
            mode_tab="preset_mode",
            temperature=0.7,
            max_chars_chunk=256,
            tts=tts,
            current_backbone="VieNeu-TTS-v3-Turbo (Thử nghiệm)",
            model_loaded=True,
            stop_event=stop,
        )
    )
    assert outs[-1][0] is None
    assert "Khối phụ đề 1" in outs[-1][1] or "Khối phụ đề" in outs[-1][1]


def test_stop_event_after_block_1_aborts_cleanly():
    tts = StubTTS()
    stop = threading.Event()
    gen = synthesize_srt_handler(
        HAPPY_SRT,
        voice_choice="DefaultVoice",
        custom_audio=None,
        custom_text="",
        mode_tab="preset_mode",
        temperature=0.7,
        max_chars_chunk=256,
        tts=tts,
        current_backbone="VieNeu-TTS-v3-Turbo (Thử nghiệm)",
        model_loaded=True,
        stop_event=stop,
    )
    outs = []
    for ev in gen:
        outs.append(ev)
        if "Khối 1" in (ev[1] or ""):
            stop.set()
    assert outs[-1][0] is None
    assert "⏹️" in outs[-1][1]


def test_model_not_loaded():
    stop = threading.Event()
    outs = _drive(
        synthesize_srt_handler(
            HAPPY_SRT,
            voice_choice="DefaultVoice",
            custom_audio=None,
            custom_text="",
            mode_tab="preset_mode",
            temperature=0.7,
            max_chars_chunk=256,
            tts=None,
            current_backbone="",
            model_loaded=False,
            stop_event=stop,
        )
    )
    assert outs == [(None, "⚠️ Vui lòng tải model trước!")]
