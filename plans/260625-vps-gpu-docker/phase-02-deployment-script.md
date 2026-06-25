---
phase: 2
title: "Deployment Script"
status: pending
priority: P1
dependencies: [1]
---

# Phase 2: Deployment Script

## Overview

Create a one-command deployment script for GPU VPS. User clones repo, runs script, gets Gradio Web UI with SRT tab at `http://<vps-ip>:7860`.

## Requirements

- Functional: Single script handles prereq check, build, and run
- Non-functional: Idempotent (safe to re-run), handles existing containers

## Related Code Files

- Create: `deploy-vps.sh` — deployment script at project root

## Implementation Steps

1. **Create `deploy-vps.sh`** with:
   - Check NVIDIA driver + Container Toolkit (`nvidia-smi`, `docker info | grep -i nvidia`)
   - Check Docker + Docker Compose installed
   - Copy `.env.example` to `.env` if not exists
   - Build image: `docker compose -f docker/docker-compose.yml --profile gpu build`
   - Stop existing container if running
   - Start: `docker compose -f docker/docker-compose.yml --profile gpu up -d`
   - Print access URL with detected IP
   - Print log tail command for reference

2. **Script should handle:**
   - First-time setup (no image, no container)
   - Rebuild after git pull (re-build + restart)
   - HuggingFace token prompt if `HF_TOKEN` not set in `.env`

## Success Criteria

- [ ] `deploy-vps.sh` exists and is executable
- [ ] Script checks prerequisites before building
- [ ] Script is idempotent
- [ ] Output shows access URL after successful deploy
