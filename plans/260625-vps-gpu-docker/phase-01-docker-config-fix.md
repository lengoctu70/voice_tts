---
phase: 1
title: "Docker Config Fix"
status: pending
priority: P1
dependencies: []
---

# Phase 1: Docker Config Fix

## Overview

Fix `docker-compose.yml` GPU profile to use `prod` target and optimize `.dockerignore` to exclude non-runtime files from the build context.

## Requirements

- Functional: GPU Docker build must include SRT code and GPU deps (`--group gpu`)
- Non-functional: Build context should be minimal (exclude tests, docs, plans, finetune)

## Related Code Files

- Modify: `docker/docker-compose.yml` — change GPU service build target from `dev` to `prod`
- Modify: `.dockerignore` — add non-runtime directories

## Implementation Steps

1. **Fix `docker/docker-compose.yml`** — GPU service:
   - Change `target: dev` to `target: prod`
   - This ensures `uv sync --no-dev --group gpu` runs (includes PyTorch, CUDA deps)

2. **Update `.dockerignore`** — add:
   ```
   plans/
   .claude/
   docs/
   tests/
   finetune/
   client/
   examples/
   *.md
   !README.md
   !README_PYPI.md
   Makefile
   *.bat
   ```

## Success Criteria

- [ ] `docker-compose.yml` GPU profile uses `target: prod`
- [ ] `.dockerignore` excludes non-runtime directories
- [ ] `docker build -f docker/Dockerfile.gpu --target prod .` succeeds (dry run syntax check)
