---
phase: 3
title: "Verification"
status: pending
priority: P2
dependencies: [1, 2]
---

# Phase 3: Verification

## Overview

Verify Docker build succeeds locally (syntax/structure only — GPU test requires VPS). Ensure SRT files are included in the image and not excluded by `.dockerignore`.

## Requirements

- Functional: Confirm all SRT source files land in the built image
- Non-functional: Build context size is reasonable (no bloat from tests/docs/plans)

## Related Code Files

- Read: `docker/Dockerfile.gpu`
- Read: `.dockerignore`
- Read: `src/vieneu_utils/srt_*.py`
- Read: `apps/srt_ui_handler.py`

## Implementation Steps

1. **Dry-run build check** (no GPU needed):
   ```bash
   docker build -f docker/Dockerfile.gpu --target prod --no-cache --progress=plain . 2>&1 | head -50
   ```
   Confirm COPY step includes `src/` and `apps/`.

2. **Verify SRT files not excluded**:
   ```bash
   # Build context should include these:
   # src/vieneu_utils/srt_parser.py
   # src/vieneu_utils/srt_to_audio.py
   # src/vieneu_utils/srt_audio_ops.py
   # apps/srt_ui_handler.py
   ```

3. **Verify excluded dirs**:
   ```bash
   # These should NOT be in the image:
   # plans/, .claude/, docs/, tests/, finetune/, client/
   ```

4. **On VPS (manual, post-deploy)**:
   - `docker exec <container> ls /workspace/src/vieneu_utils/srt_parser.py`
   - Open `http://<vps-ip>:7860` → confirm SRT tab exists
   - Paste test SRT → generate audio → confirm WAV output

## Success Criteria

- [ ] Docker build completes without error
- [ ] SRT source files present in image
- [ ] Non-runtime dirs excluded from image
- [ ] `deploy-vps.sh` runs end-to-end on VPS (manual verification)
