"""Hub-aware utilities for training scripts running on the HF Space.

Provides checkpoint resume from Hub, retried uploads, and trackio init.
Imported by sft.py, train.py, and eval.py — no heavy ML imports at module level.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("flight_rebooking.space_runner")

# ─────────────────────────────────────────────────────────────
# Hub repo constants
# ─────────────────────────────────────────────────────────────

HUB_REPOS = {
    "sft": "Vanshar0ra/irrops-sft-adapter",
    "grpo-p1": "Vanshar0ra/irrops-grpo-phase1",
    "grpo-p2": "Vanshar0ra/irrops-grpo-phase2",
    "gold": "Vanshar0ra/irrops-gold-trajectories",
}


# ─────────────────────────────────────────────────────────────
# trackio initialisation
# ─────────────────────────────────────────────────────────────


def init_trackio(run_name: str, config: Optional[dict] = None) -> None:
    """Initialise a trackio run and log GPU info + config.

    Safe to call even when trackio isn't available — logs a warning and
    returns silently.
    """
    try:
        import trackio
    except ImportError:
        logger.warning("trackio not installed — skipping init")
        return

    try:
        import torch

        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
            "gpu_count": torch.cuda.device_count(),
            "cuda_version": torch.version.cuda or "n/a",
        }
    except Exception:
        gpu_info = {"gpu_name": "unknown", "gpu_count": 0, "cuda_version": "n/a"}

    merged_config = {**gpu_info, **(config or {})}

    try:
        trackio.init(run_name=run_name, config=merged_config)
        logger.info("trackio run '%s' initialised", run_name)
    except Exception as exc:
        logger.warning("trackio init failed: %s", exc)


# ─────────────────────────────────────────────────────────────
# Hub checkpoint resume
# ─────────────────────────────────────────────────────────────


def check_hub_resume(
    hub_repo_id: str,
    local_dir: str = "outputs/resume",
) -> Optional[str]:
    """Check Hub for an existing checkpoint and download if found.

    Returns the local path to the latest checkpoint directory, or None if
    no checkpoint exists on Hub.
    """
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files

    api = HfApi()
    try:
        files = list_repo_files(hub_repo_id)
    except Exception:
        logger.info("No existing repo at %s — starting fresh", hub_repo_id)
        return None

    # Look for checkpoint directories (e.g. checkpoint-25, checkpoint-50)
    checkpoint_dirs = set()
    for f in files:
        parts = f.split("/")
        for part in parts:
            if part.startswith("checkpoint-"):
                checkpoint_dirs.add(part)

    if not checkpoint_dirs:
        logger.info("Repo %s exists but has no checkpoints", hub_repo_id)
        return None

    # Pick the highest-numbered checkpoint
    def _step_num(name: str) -> int:
        try:
            return int(name.split("-")[1])
        except (IndexError, ValueError):
            return 0

    latest = max(checkpoint_dirs, key=_step_num)
    logger.info("Found checkpoint %s in %s — downloading", latest, hub_repo_id)

    local_path = Path(local_dir) / latest
    local_path.mkdir(parents=True, exist_ok=True)

    # Download all files in that checkpoint directory
    for f in files:
        if f.startswith(latest + "/") or f.startswith(f"{latest}/"):
            try:
                hf_hub_download(
                    repo_id=hub_repo_id,
                    filename=f,
                    local_dir=local_dir,
                )
            except Exception as exc:
                logger.warning("Failed to download %s: %s", f, exc)

    return str(local_path)


# ─────────────────────────────────────────────────────────────
# Retried push-to-hub wrapper
# ─────────────────────────────────────────────────────────────


def push_with_retry(
    fn: Any,
    *args: Any,
    max_retries: int = 5,
    **kwargs: Any,
) -> Any:
    """Call *fn* with exponential backoff on transient Hub errors.

    Typical usage:
        push_with_retry(model.push_to_hub, "Vanshar0ra/my-model")
    """
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            err_str = str(exc)
            retryable = any(
                code in err_str
                for code in ("429", "503", "504", "Connection", "Timeout")
            )
            if retryable and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    "push_with_retry attempt %d/%d failed (%s) — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    err_str[:120],
                    wait,
                )
                time.sleep(wait)
            else:
                raise
