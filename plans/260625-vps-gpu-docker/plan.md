---
title: "VPS GPU Docker deployment with SRT"
description: "Fix Docker config to build custom image with SRT pipeline, add deployment script for GPU VPS, optimize .dockerignore"
status: pending
priority: P1
branch: "main"
tags: ["docker", "gpu", "vps", "deployment"]
blockedBy: []
blocks: []
created: "2026-06-25T08:25:51.991Z"
createdBy: "ck:plan"
source: skill
---

# VPS GPU Docker deployment with SRT

## Overview

Deploy the forked VieNeu-TTS (with custom SRT-to-Audio pipeline) on a rented GPU VPS via Docker. The existing Docker infra is mostly correct but needs minor fixes: the `docker-compose.yml` GPU dev profile uses target `dev` (CPU-only `uv sync`), `.dockerignore` doesn't exclude non-runtime directories, and there's no one-command deployment script.

## Current State

- `Dockerfile.gpu` has two stages: `dev` (CPU-only) and `prod` (GPU + full app)
- `docker-compose.yml` GPU profile points to `target: dev` — **wrong for production**
- `docker-compose.build.yml` correctly uses `target: prod` but is build-only
- `docker-compose.prod.yml` uses pre-built image, no build step
- SRT code lives in `src/vieneu_utils/srt_*.py` + `apps/srt_ui_handler.py` — all inside `COPY . .` in Dockerfile, no exclusion in `.dockerignore`
- `.dockerignore` missing: `plans/`, `.claude/`, `docs/`, `tests/`, `finetune/`, `client/`

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Docker Config Fix](./phase-01-docker-config-fix.md) | Pending |
| 2 | [Deployment Script](./phase-02-deployment-script.md) | Pending |
| 3 | [Verification](./phase-03-verification.md) | Pending |

## Dependencies

- Requires NVIDIA GPU VPS with CUDA >= 12.8 and NVIDIA Container Toolkit
- SRT plan `260625-srt-to-audio` is completed; SRT code is in the codebase
