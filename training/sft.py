#!/usr/bin/env python3
"""Supervised fine-tuning on gold-standard trajectories.

Loads gold trajectories from the Hub (Vanshar0ra/irrops-gold-trajectories)
or falls back to local training/data/gold_trajectories.jsonl.  Uses TRL's
SFTTrainer with chat-template formatting on Unsloth-quantised Qwen.

Usage:
    python training/sft.py
    python training/sft.py --epochs 5 --lr 1e-5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("flight_rebooking.sft")

# ─────────────────────────────────────────────────────────────
# Make repo root importable
# ─────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ─────────────────────────────────────────────────────────────
# Gold trajectory loading
# ─────────────────────────────────────────────────────────────

LOCAL_GOLD_PATH = Path(__file__).resolve().parent / "data" / "gold_trajectories.jsonl"


def _load_gold_from_hub(repo_id: str) -> list[dict] | None:
    """Try to load gold trajectories from Hub dataset repo."""
    try:
        from huggingface_hub import hf_hub_download

        local_file = hf_hub_download(
            repo_id=repo_id,
            filename="gold_trajectories.jsonl",
            repo_type="dataset",
        )
        rows = []
        with open(local_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        logger.info("Loaded %d trajectories from Hub (%s)", len(rows), repo_id)
        return rows
    except Exception as exc:
        logger.warning("Could not load from Hub (%s): %s", repo_id, exc)
        return None


def _load_gold_local(path: Path) -> list[dict]:
    """Load gold trajectories from local JSONL file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Gold trajectories not found at {path}. "
            f"Run training/gold_gen.py first, or push to Hub."
        )
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    logger.info("Loaded %d trajectories from %s", len(rows), path)
    return rows


def load_gold_trajectories(hub_repo_id: str) -> list[dict]:
    """Load gold trajectories — Hub first, local fallback."""
    rows = _load_gold_from_hub(hub_repo_id)
    if rows is not None and len(rows) > 0:
        return rows
    return _load_gold_local(LOCAL_GOLD_PATH)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────


def main() -> None:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    from training.config import build_model_and_tokenizer
    from training.space_runner import (
        HUB_REPOS,
        check_hub_resume,
        init_trackio,
    )

    parser = argparse.ArgumentParser(description="SFT on gold trajectories")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--output-dir", type=str, default="outputs/sft", help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

    # ── Init trackio ────────────────────────────────────────
    init_trackio("sft", config={
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
    })

    # ── Load gold trajectories ──────────────────────────────
    hub_repo = HUB_REPOS["gold"]
    gold_rows = load_gold_trajectories(hub_repo)
    print(f"Loaded {len(gold_rows)} gold trajectories")

    # Each row has a "messages" field: list of {role, content} dicts
    # The SFTTrainer with chat template formatting needs a Dataset with a "messages" column.
    dataset = Dataset.from_list([{"messages": row["messages"]} for row in gold_rows])
    print(f"Dataset: {len(dataset)} examples")

    # ── Load model and tokenizer ────────────────────────────
    print("Loading model and tokenizer...")
    model, tokenizer = build_model_and_tokenizer()

    # ── Check for Hub resume ────────────────────────────────
    sft_repo = HUB_REPOS["sft"]
    resume_path = check_hub_resume(sft_repo, local_dir=args.output_dir)
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")

    # ── SFT config ──────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_steps=25,
        save_total_limit=3,
        logging_steps=1,
        report_to="trackio",
        push_to_hub=True,
        hub_model_id=sft_repo,
        hub_strategy="every_save",
        bf16=True,
        max_seq_length=4096,
    )

    # ── Trainer ─────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_config,
    )

    # ── Train ───────────────────────────────────────────────
    print("Starting SFT training...")
    trainer.train(resume_from_checkpoint=resume_path)

    # ── Save and push ───────────────────────────────────────
    print("Saving final model...")
    trainer.save_model()
    print(f"SFT training complete. Model saved to {args.output_dir}")

    if sft_config.push_to_hub:
        print(f"Pushing to Hub: {sft_repo}")
        trainer.push_to_hub()
        print("Push complete.")


if __name__ == "__main__":
    main()
