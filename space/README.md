---
title: IRROPS Trainer
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
hardware: t4-medium
sleep_time: 3600
suggested_storage: small
---

# IRROPS Trainer — Flight Rebooking RL Training Space

Training container for the flight rebooking RL agent. Runs SFT, GRPO, and evaluation jobs on a T4 medium GPU.

## JOB Values

Set the `JOB` environment variable in Space Settings to select the training job.

| JOB | Description | Hub Repo (push target) | Approx. Runtime | Approx. Cost |
|-----|-------------|----------------------|-----------------|--------------|
| `idle` | No training — just trackio dashboard | — | Indefinite | $0/hr (pause when done) |
| `sft` | Supervised fine-tuning on gold trajectories | `Vanshar0ra/irrops-sft-adapter` | ~30 min | ~$0.45 |
| `grpo-smoke` | GRPO smoke test (2 easy, 2 steps) | (no push) | ~10 min | ~$0.15 |
| `grpo-p1` | GRPO Phase 1 (50 easy, 200 steps) | `Vanshar0ra/irrops-grpo-phase1` | ~2-3 hrs | ~$2-3 |
| `grpo-p2` | GRPO Phase 2 (easy+med+hard, 400 steps) | `Vanshar0ra/irrops-grpo-phase2` | ~4-6 hrs | ~$4-5 |
| `eval` | Evaluate SFT + Phase1 + Phase2 checkpoints | — (logs to trackio) | ~30-60 min | ~$0.50-1 |

## Hub Repos

| Repo | Contents |
|------|----------|
| `Vanshar0ra/irrops-gold-trajectories` | Gold JSONL trajectories (input for SFT) |
| `Vanshar0ra/irrops-sft-adapter` | SFT LoRA adapter (input for GRPO) |
| `Vanshar0ra/irrops-grpo-phase1` | GRPO Phase 1 adapter |
| `Vanshar0ra/irrops-grpo-phase2` | GRPO Phase 2 adapter (final model) |

## Monitoring

- **Trackio dashboard**: Visit the Space URL — live loss/reward plots
- **Logs tab**: Click "Logs" on the Space page for stdout/stderr

## Panic Button

If a job is stuck, consuming budget, or producing bad results:

1. **Quick**: Go to Space Settings → set `JOB=idle` → click "Restart"
2. **Full stop**: Click "Pause" on the Space UI — billing stops immediately

## Required Secrets & Variables

Set these in Space Settings → Variables and secrets:

| Type | Key | Value |
|------|-----|-------|
| Secret | `HF_TOKEN` | HF write token (for push-to-hub) |
| Variable | `JOB` | One of: `idle`, `sft`, `grpo-smoke`, `grpo-p1`, `grpo-p2`, `eval` |
| Variable | `PROJECT_REF` | Git tag/branch to build from (e.g., `space-v1`) |
| Variable | `HF_HUB_ENABLE_HF_TRANSFER` | `1` (fast uploads) |
