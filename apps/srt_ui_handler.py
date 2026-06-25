"""Gradio handler for the SRT-to-Audio tab.

Kept in its own module so it can be unit-tested without importing
the full Gradio app graph. `gradio_main.synthesize_srt_handler`
delegates here, passing the currently loaded TTS engine.
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
from typing import Callable, Optional

import numpy as np
import soundfile as sf

from vieneu_utils.srt_parser import SrtValidationError, parse_srt
from vieneu_utils.srt_to_audio import (
    MAX_TOTAL_DURATION_S,
    SrtGenerationError,
    format_chunk_log,
    synthesize_srt,
)


def _resolve_reference(
    tts,
    current_backbone: str,
    voice_choice: str,
    custom_audio,
    custom_text: str,
    mode_tab: str,
    voice_resolver: Callable[[str], str] | None = None,
):
    """Mirrors the reference resolution from `synthesize_speech` in
    `apps/gradio_main.py`. Returns (ref_codes, ref_text_raw,
    v3_voice_token_id, resolved_voice_id)."""
    ref_codes = None
    ref_text_raw = ""
    v3_voice_token_id = None
    v_id = None

    if mode_tab == "preset_mode":
        if not voice_choice:
            raise ValueError("Vui lòng chọn giọng mẫu.")
        if "⚠️" in (voice_choice or ""):
            raise ValueError(
                "Không có giọng mẫu khả dụng. Vui lòng chuyển sang Tab Voice Cloning."
            )
        v_id = voice_resolver(voice_choice) if voice_resolver else voice_choice
        voice_data = tts.get_preset_voice(v_id)
        ref_codes = voice_data["codes"]
        ref_text_raw = voice_data.get("text", "")
        v3_voice_token_id = voice_data.get("reserved_id")

    elif mode_tab == "custom_mode":
        if custom_audio is None:
            raise ValueError("Vui lòng upload file Audio mẫu (Reference Audio)!")
        cb_lower = (current_backbone or "").lower()
        needs_ref_text = "v2-turbo" not in cb_lower and "v3" not in cb_lower
        if needs_ref_text and (not custom_text or not custom_text.strip()):
            raise ValueError(
                "Vui lòng nhập nội dung văn bản của Audio mẫu (Reference Text)!"
            )
        ref_text_raw = custom_text.strip() if custom_text else ""
        ref_codes = tts.encode_reference(custom_audio)

    else:
        raise ValueError(f"Chế độ tham chiếu không hợp lệ: {mode_tab!r}")

    if "torch" in sys.modules:
        import torch

        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()

    return ref_codes, ref_text_raw, v3_voice_token_id, v_id


def _make_chunk_synthesizer(
    tts,
    current_backbone: str,
    *,
    ref_codes,
    ref_text_raw: str,
    v3_voice_token_id,
    voice_id: Optional[str],
    temperature: float,
    max_chars_chunk: int,
):
    """Return a `synthesize_chunk(text) -> np.ndarray` closure that mirrors
    the per-utterance call signature used by `synthesize_speech` for the
    currently loaded backbone family. One block at a time → no batching."""
    cb_lower = (current_backbone or "").lower()
    is_v3 = "v3" in cb_lower
    is_v2_turbo = "v2-turbo" in cb_lower

    def synth(text: str) -> np.ndarray:
        if is_v3:
            kwargs = {
                "temperature": temperature,
                "max_chars": max_chars_chunk,
            }
            if v3_voice_token_id is not None and voice_id is not None:
                kwargs["voice"] = voice_id
            else:
                kwargs["ref_codes"] = ref_codes
            wav = tts.infer(text, **kwargs)
        elif is_v2_turbo:
            wav = tts.infer(
                text,
                ref_codes=ref_codes,
                temperature=temperature,
                max_chars=max_chars_chunk,
            )
        else:
            wav = tts.infer(
                text,
                ref_codes=ref_codes,
                ref_text=ref_text_raw,
                temperature=temperature,
                max_chars=max_chars_chunk,
            )
        return np.asarray(wav, dtype=np.float32)

    return synth


def synthesize_srt_handler(
    srt_text: str,
    voice_choice: str,
    custom_audio,
    custom_text: str,
    mode_tab: str,
    temperature: float,
    max_chars_chunk: int,
    *,
    tts,
    current_backbone: str,
    model_loaded: bool,
    stop_event: threading.Event,
    voice_resolver: Callable[[str], str] | None = None,
):
    """Generator producing `(audio_filepath_or_None, status_message)` tuples
    for the Gradio audio + status outputs.

    On invalid SRT → first yield is `(None, "❌ …")` and tts.infer is never
    called. On silent-output failure → final yield references the failing
    block index, no file produced. On `stop_event.set()` mid-run → yields a
    `⏹️ Dừng…` message and returns without a final file.
    """
    stop_event.clear()

    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"
        return

    if not srt_text or not srt_text.strip():
        yield None, "⚠️ Vui lòng nhập nội dung SRT!"
        return

    try:
        validated_blocks = parse_srt(srt_text)
    except SrtValidationError as e:
        yield None, f"❌ SRT không hợp lệ: {e.message}"
        return

    if validated_blocks and validated_blocks[-1].end_s > MAX_TOTAL_DURATION_S:
        yield None, (
            f"❌ Tổng thời lượng SRT vượt giới hạn "
            f"({validated_blocks[-1].end_s:.1f}s > {MAX_TOTAL_DURATION_S:.0f}s)."
        )
        return

    yield None, "📄 Đang xử lý reference…"
    try:
        ref_codes, ref_text_raw, v3_voice_token_id, voice_id = _resolve_reference(
            tts=tts,
            current_backbone=current_backbone,
            voice_choice=voice_choice,
            custom_audio=custom_audio,
            custom_text=custom_text,
            mode_tab=mode_tab,
            voice_resolver=voice_resolver,
        )
    except Exception as e:
        yield None, f"❌ Lỗi xử lý reference: {e}"
        return

    sr = int(getattr(tts, "sample_rate", 24000) or 24000)
    synth_chunk = _make_chunk_synthesizer(
        tts=tts,
        current_backbone=current_backbone,
        ref_codes=ref_codes,
        ref_text_raw=ref_text_raw,
        v3_voice_token_id=v3_voice_token_id,
        voice_id=voice_id,
        temperature=temperature,
        max_chars_chunk=max_chars_chunk,
    )

    t0 = time.time()
    yield None, "🚀 Bắt đầu tổng hợp từ SRT…"

    gen = synthesize_srt(
        srt_text,
        synthesize_chunk=synth_chunk,
        sr=sr,
        should_stop=lambda: stop_event.is_set(),
    )

    final = None
    try:
        while True:
            try:
                ev = next(gen)
            except StopIteration as stop_iter:
                final = stop_iter.value
                break
            kind, msg = ev[0], ev[1]
            yield None, msg
    except SrtValidationError as e:
        yield None, f"❌ SRT không hợp lệ: {e.message}"
        return
    except SrtGenerationError as e:
        yield None, f"❌ Khối phụ đề {e.block}: {e}"
        return
    except Exception as e:
        yield None, f"❌ Lỗi tổng hợp SRT: {e}"
        return

    wav, logs = final
    if stop_event.is_set() and any(l.status == "cancelled" for l in logs):
        yield None, "⏹️ Đã dừng tạo giọng nói."
        return

    if wav is None or len(wav) == 0:
        yield None, "❌ Không sinh được audio nào."
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, wav, sr)
        out_path = tmp.name

    n = len(logs)
    cut_count = sum(1 for l in logs if l.status == "cut")
    retry_count = sum(l.retries for l in logs)
    dt = time.time() - t0
    summary = (
        f"✅ Hoàn tất! {n} khối, {cut_count} cắt, {retry_count} retry, "
        f"sr={sr}Hz, thời gian={dt:.2f}s"
    )
    yield out_path, summary
