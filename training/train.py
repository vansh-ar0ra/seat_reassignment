#!/usr/bin/env python3
"""GRPO training entry point with smoke / phase-1 / phase-2 modes.

Usage:
    python training/train.py --smoke          # 8 easy, 2 steps — sanity check
    python training/train.py --phase 1        # 200 easy, 100 steps → push grpo-phase1
    python training/train.py --phase 2        # 50 easy + 100 med + 100 hard, 400 steps → push grpo-phase2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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
        "gradient_accumulation_steps": 2,  # gen_batch = 1×1×2 = 2 → divisible by num_generations
        "num_generations": 2,              # minimum for GRPO advantage; keeps smoke cheap
        "init_from": "sft",
        "push_to": None,  # no push for smoke
        "output_dir": "outputs/grpo-smoke",
    },
    "phase1": {
        "n_easy": 200,
        "n_medium": 0,
        "n_hard": 0,
        "max_steps": 100,
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
    from trl import GRPOTrainer

    from training.config import build_grpo_config, build_model_and_tokenizer
    from training.dataset import build_dataset
    from training.rewards import REWARD_FUNCS
    from training.rollout import rollout_func
    from training.space_runner import (
        HUB_REPOS,
        check_hub_resume,
        init_trackio,
    )

    # ── Monkey-patch shuffle_sequence_dict to dump dict before crash ──
    import trl.trainer.utils as _trl_utils
    import trl.trainer.grpo_trainer as _grpo_mod

    _orig_shuffle = _trl_utils.shuffle_sequence_dict

    def _debug_shuffle(seq_dict):
        import torch as _torch
        print("=" * 70)
        print("[SHUFFLE DEBUG] dict at shuffle time:")
        # Determine expected batch size from majority of tensors
        batch_size = None
        for k, v in seq_dict.items():
            if hasattr(v, 'shape') and v.ndim >= 1:
                if batch_size is None:
                    batch_size = v.shape[0]
                elif v.shape[0] > batch_size:
                    batch_size = v.shape[0]

        for k, v in seq_dict.items():
            if v is None:
                print(f"  {k}: None")
            elif hasattr(v, 'shape'):
                print(f"  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
            elif hasattr(v, '__len__'):
                try:
                    inner = type(v[0]).__name__ if len(v) > 0 else "empty"
                    print(f"  {k}: {type(v).__name__} len={len(v)} inner={inner}")
                except Exception as e:
                    print(f"  {k}: {type(v).__name__} (introspection failed: {e})")
            else:
                print(f"  {k}: {type(v).__name__} value={v!r}")

        print("=" * 70)
        return _orig_shuffle(seq_dict)

    _trl_utils.shuffle_sequence_dict = _debug_shuffle
    _grpo_mod.shuffle_sequence_dict = _debug_shuffle

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

    import os
    print(f"[ENV] CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}")
    print(f"[ENV] TORCH_USE_CUDA_DSA={os.environ.get('TORCH_USE_CUDA_DSA')}")

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

    # Vocab-size diagnostic — helps debug embedding OOB errors
    print(f"[VOCAB CHECK] model.config.vocab_size={model.config.vocab_size}")
    print(f"[VOCAB CHECK] len(tokenizer)={len(tokenizer)}")
    print(f"[VOCAB CHECK] tokenizer.vocab_size={tokenizer.vocab_size}")

    # If we have an SFT or phase1 adapter on Hub, load those weights
    if init_repo:
        try:
            from peft import PeftModel

            print(f"Loading adapter from Hub: {init_repo}")
            model = PeftModel.from_pretrained(model, init_repo, is_trainable=True)
            print("Adapter loaded (trainable).")
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
    if "num_generations" in mode:
        overrides["num_generations"] = mode["num_generations"]
    if push_repo:
        overrides["push_to_hub"] = True
        overrides["hub_model_id"] = push_repo
        overrides["hub_strategy"] = "every_save"
    else:
        overrides["push_to_hub"] = False

    grpo_config = build_grpo_config(**overrides)

    # ── Sanity-check config ──────────────────────────────────
    print(f"[CONFIG] per_device_train_batch_size={grpo_config.per_device_train_batch_size}")
    print(f"[CONFIG] gradient_accumulation_steps={grpo_config.gradient_accumulation_steps}")
    print(f"[CONFIG] num_generations={grpo_config.num_generations}")
    print(f"[CONFIG] generation_batch_size={grpo_config.generation_batch_size}")

    # ── Build trainer ───────────────────────────────────────
    # Stock TRL 0.29+ GRPOTrainer with environment rollout support.
    trainer = GRPOTrainer(
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
    # Save adapter to disk FIRST — don't lose work to a failed push
    local_adapter_path = Path(mode["output_dir"]) / "final_adapter"
    print(f"Saving final adapter to {local_adapter_path} ...")
    trainer.save_model(str(local_adapter_path))
    print(f"Adapter saved to disk: {local_adapter_path}")

    if push_repo and grpo_config.push_to_hub:
        print(f"Pushing to Hub: {push_repo}")
        from training.space_runner import push_with_retry
        push_with_retry(trainer.push_to_hub)
        print("Push complete.")

    # ── Trackio phase-boundary marker ────────────────────────
    try:
        import trackio
        trackio.log({
            f"phase/{mode_key}/completed": 1,
            f"phase/{mode_key}/final_step": mode["max_steps"],
        })
    except Exception:
        pass


if __name__ == "__main__":
    main()
