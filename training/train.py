#!/usr/bin/env python3
"""GRPO training entry point with smoke / phase-1 / phase-2 modes.

Usage:
    python training/train.py --smoke          # 2 easy, 2 steps — sanity check
    python training/train.py --phase 1        # 50 easy, 200 steps → push grpo-phase1
    python training/train.py --phase 2        # 50 easy + 100 med + 100 hard, 400 steps → push grpo-phase2
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Single-GPU: TRL's GRPOTrainer uses vLLM with distributed_executor_backend='external_launcher',
# which expects these env vars (normally set by torchrun/accelerate).
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

logger = logging.getLogger("flight_rebooking.train")

# ─────────────────────────────────────────────────────────────
# Make repo root importable
# ─────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ─────────────────────────────────────────────────────────────
# Mode definitions
# ─────────────────────────────────────────────────────────────

MODES = {
    "smoke": {
        "n_easy": 8,
        "n_medium": 0,
        "n_hard": 0,
        "max_steps": 2,
        "save_steps": 1,
        "gradient_accumulation_steps": 1,  # small dataset — keep total batch ≤ n_easy
        "init_from": "sft",
        "push_to": None,  # no push for smoke
        "output_dir": "outputs/grpo-smoke",
    },
    "phase1": {
        "n_easy": 50,
        "n_medium": 0,
        "n_hard": 0,
        "max_steps": 200,
        "save_steps": 25,
        "init_from": "sft",
        "push_to": "grpo-p1",
        "output_dir": "outputs/grpo-phase1",
    },
    "phase2": {
        "n_easy": 50,
        "n_medium": 100,
        "n_hard": 100,
        "max_steps": 400,
        "save_steps": 25,
        "init_from": "grpo-p1",
        "push_to": "grpo-p2",
        "output_dir": "outputs/grpo-phase2",
    },
}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────


def main() -> None:
    from training.config import build_grpo_config, build_model_and_tokenizer
    from training.dataset import build_dataset
    from training.env_grpo_trainer import EnvGRPOTrainer
    from training.rewards import REWARD_FUNCS
    from training.rollout import rollout_func
    from training.space_runner import (
        HUB_REPOS,
        check_hub_resume,
        init_trackio,
    )

    parser = argparse.ArgumentParser(description="GRPO training")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true", help="Smoke test: 2 easy, 2 steps")
    group.add_argument("--phase", type=int, choices=[1, 2], help="Training phase (1 or 2)")
    args = parser.parse_args()

    # Resolve mode
    if args.smoke:
        mode_key = "smoke"
    else:
        mode_key = f"phase{args.phase}"
    mode = MODES[mode_key]

    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

    print(f"Mode: {mode_key}")
    print(f"  Dataset: easy={mode['n_easy']}, medium={mode['n_medium']}, hard={mode['n_hard']}")
    print(f"  Max steps: {mode['max_steps']}")
    print(f"  Init from: {mode['init_from']}")
    print(f"  Push to: {mode['push_to'] or '(none)'}")

    # ── Init trackio ────────────────────────────────────────
    init_trackio(f"grpo-{mode_key}", config={
        "mode": mode_key,
        "max_steps": mode["max_steps"],
        "n_easy": mode["n_easy"],
        "n_medium": mode["n_medium"],
        "n_hard": mode["n_hard"],
    })

    # ── Build dataset ───────────────────────────────────────
    dataset = build_dataset(
        n_easy=mode["n_easy"],
        n_medium=mode["n_medium"],
        n_hard=mode["n_hard"],
    )
    print(f"Dataset: {len(dataset)} samples")

    # ── Load model ──────────────────────────────────────────
    # For phase2, load from phase1 checkpoint on Hub; for smoke/phase1, load from SFT adapter
    init_key = mode["init_from"]
    init_repo = HUB_REPOS.get(init_key)

    print(f"Loading model (init from: {init_key} @ {init_repo})...")
    model, tokenizer = build_model_and_tokenizer()

    # If we have an SFT or phase1 adapter on Hub, load those weights
    if init_repo:
        try:
            from peft import PeftModel

            print(f"Loading adapter from Hub: {init_repo}")
            model = PeftModel.from_pretrained(model, init_repo)
            model = model.merge_and_unload()
            print("Adapter merged successfully.")
            # Re-apply LoRA for GRPO training
            from training.config import DEFAULT_LORA_RANK
            from unsloth import FastLanguageModel

            model = FastLanguageModel.get_peft_model(
                model,
                r=DEFAULT_LORA_RANK,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_alpha=DEFAULT_LORA_RANK,
                lora_dropout=0,
                use_gradient_checkpointing="unsloth",
                random_state=42,
            )
        except Exception as exc:
            logger.warning(
                "Could not load adapter from %s (%s) — training from base model",
                init_repo, exc,
            )

    # ── Check for Hub resume (of the target repo) ──────────
    push_key = mode["push_to"]
    push_repo = HUB_REPOS.get(push_key) if push_key else None
    resume_path = None
    if push_repo:
        resume_path = check_hub_resume(push_repo, local_dir=mode["output_dir"])
        if resume_path:
            print(f"Resuming from checkpoint: {resume_path}")

    # ── Build GRPO config ───────────────────────────────────
    overrides = {
        "max_steps": mode["max_steps"],
        "save_steps": mode["save_steps"],
        "output_dir": mode["output_dir"],
    }
    if "gradient_accumulation_steps" in mode:
        overrides["gradient_accumulation_steps"] = mode["gradient_accumulation_steps"]
    if push_repo:
        overrides["push_to_hub"] = True
        overrides["hub_model_id"] = push_repo
        overrides["hub_strategy"] = "every_save"
    else:
        overrides["push_to_hub"] = False

    grpo_config = build_grpo_config(**overrides)

    # ── Build trainer ───────────────────────────────────────
    # EnvGRPOTrainer subclasses TRL 0.19's GRPOTrainer and overrides
    # _generate_and_score_completions to drive environment rollouts.
    trainer = EnvGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=REWARD_FUNCS,
        rollout_func=rollout_func,
    )

    # ── Train ───────────────────────────────────────────────
    print(f"Starting GRPO training ({mode_key})...")
    trainer.train(resume_from_checkpoint=resume_path)

    # ── Save and push ───────────────────────────────────────
    print("Saving final model...")
    trainer.save_model()
    print(f"Training complete. Model saved to {mode['output_dir']}")

    if push_repo and grpo_config.push_to_hub:
        print(f"Pushing to Hub: {push_repo}")
        trainer.push_to_hub()
        print("Push complete.")


if __name__ == "__main__":
    main()
