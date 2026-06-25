---
phase: 4
title: "Gradio UI tab integration"
status: completed
effort: "M"
---

# Phase 4: Gradio UI tab integration

## Overview

Wire the orchestrator from Phase 3 into `apps/gradio_main.py` as a third tab "📜 SRT → Audio", reusing the currently loaded model, selected voice/cloning reference, temperature, and chunk settings. Additive only — no edits to existing single-speaker or conversation handlers.

## Files

- Modify `apps/gradio_main.py`:
  - Add new `gr.Tab("📜 SRT → Audio", id="srt_tab", visible=False) as srt_tab` inside `main_input_tabs`.
  - Inside the tab: `srt_input = gr.Textbox(label="Nội dung SRT", lines=14, placeholder=...)`, `btn_generate_srt = gr.Button(...)`, `srt_log_md = gr.Markdown(...)`.
  - Add `synthesize_srt_handler(...)` generator paralleling `synthesize_speech` for reference handling, but delegating per-chunk synthesis to `vieneu_utils.srt_to_audio.synthesize_srt`.
  - Register the click event; reuse `btn_stop` via `_STOP_EVENT`; toggle `srt_tab` visibility in `on_backbone_change` exactly like `conv_tab` (visible whenever the loaded model supports single-speaker synthesis — which is all of them).
- Modify `apps/ui_constants.py` (optional): add a default SRT example string so the new textbox has a discoverable placeholder.

No changes to existing handlers, button outputs lists, or shared state beyond appending the new button/output controls to `restore_ui_state` / `load_model` return tuples (mirroring how `btn_generate_conv` was added).

## TDD — Tests first

UI is Gradio so heavy E2E is out of scope; rely on:

1. `tests/test_srt_ui_handler.py` — calls the handler function directly with a monkey-patched `tts` global (a minimal object exposing `sample_rate`, `infer`, `encode_reference`) and asserts:
   - For valid SRT, the handler yields at least one status string and a final `(filepath, status)` whose filepath is a `.wav` written via `soundfile.write`.
   - For invalid SRT, the first yield is `(None, "❌ …")` with the validation message and `tts.infer` is never called.
   - For silent-output failure, the final yield is `(None, "❌ Khối phụ đề …")` referencing the failing block index.
   - When `_STOP_EVENT` is set after block 1, the handler yields a "⏹️" status and returns without producing a final file.
2. Manual smoke (documented checklist, no code): load model → open SRT tab → paste 3-block fixture → click Start → verify WAV plays back with correct timing in Gradio audio component.

## Handler outline

```python
def synthesize_srt_handler(
    srt_text, voice_choice, custom_audio, custom_text, mode_tab,
    temperature, max_chars_chunk, session_id=None,
):
    global tts, current_backbone, model_loaded
    _STOP_EVENT.clear()
    if not model_loaded or tts is None:
        yield None, "⚠️ Vui lòng tải model trước!"; return
    # 1. Resolve reference (preset_mode / custom_mode) — reuse the exact block from synthesize_speech
    # 2. Build a per-chunk synthesize_chunk closure that calls tts.infer(...) with ref_codes,
    #    v3_voice_token_id, temperature, max_chars_chunk — same kwargs the existing handler uses.
    # 3. Drive the generator from synthesize_srt(...) and translate events to (None, status) yields.
    # 4. On 'done': write WAV via soundfile.write to a NamedTemporaryFile and yield (path, summary).
    # 5. On SrtValidationError / SrtGenerationError: yield (None, f"❌ {e}") and return.
```

The closure isolates the per-block synthesis call so all branch handling (v3 turbo vs v2 turbo vs standard) stays in one place that mirrors `synthesize_speech` — copy-paste the relevant single-utterance call path, omitting batching (one block at a time keeps progress reporting simple and avoids the SRT-order vs batch-order mapping problem). Batching is a follow-up.

Status string format: `Khối {i}/{N}: {format_chunk_log(log)}`. Final summary: `✅ Hoàn tất! {N} khối, {cut_count} cắt, {retry_count} retry, sr={sr}Hz`.

## Success Criteria

- [ ] Tab appears after model load; hidden before, matching `conv_tab` behavior.
- [ ] Stop button cancels generation cleanly.
- [ ] Existing single-speaker and conversation flows still work (smoke test: load model, generate one short sentence in each existing tab — no regression).
- [ ] Handler unit tests in `tests/test_srt_ui_handler.py` pass.
- [ ] No new globals; reuses `_STOP_EVENT`, `tts`, `model_loaded`, `current_backbone`.
