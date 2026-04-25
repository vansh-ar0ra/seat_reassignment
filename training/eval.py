#!/usr/bin/env python3
"""Evaluate model checkpoints across difficulty tiers.

Runs episodes using the smoke_test.run_episode() loop with a local
generate policy, comparing any combination of baseline / SFT / GRPO
Phase 1 / Phase 2 checkpoints.

Usage:
    python training/eval.py --baseline --sft
    python training/eval.py --sft --phase1 --phase2
    python training/eval.py --phase2 --episodes 10 --tiers easy medium
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("flight_rebooking.eval")

# ─────────────────────────────────────────────────────────────
# Make repo root importable
# ─────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ─────────────────────────────────────────────────────────────
# Checkpoint definitions
# ─────────────────────────────────────────────────────────────

CHECKPOINTS = {
    "baseline": {
        "description": "Base model (no fine-tuning)",
        "adapter": None,
    },
    "sft": {
        "description": "SFT adapter",
        "adapter": "Vanshar0ra/irrops-sft-adapter",
    },
    "phase1": {
        "description": "GRPO Phase 1 adapter",
        "adapter": "Vanshar0ra/irrops-grpo-phase1",
    },
    "phase2": {
        "description": "GRPO Phase 2 adapter",
        "adapter": "Vanshar0ra/irrops-grpo-phase2",
    },
}


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────


def load_model_for_eval(
    checkpoint_name: str,
    base_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    device: str = "cuda",
):
    """Load base model + optional PEFT adapter for evaluation.

    Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = CHECKPOINTS[checkpoint_name]

    print(f"  Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading base model (bfloat16)...")
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
        ).to(device)

    # Apply adapter if specified
    adapter_repo = ckpt["adapter"]
    if adapter_repo:
        try:
            from peft import PeftModel

            print(f"  Loading adapter from Hub: {adapter_repo}")
            model = PeftModel.from_pretrained(model, adapter_repo)
            print(f"  Adapter loaded: {checkpoint_name}")
        except Exception as exc:
            logger.warning("Could not load adapter %s: %s", adapter_repo, exc)
            print(f"  WARNING: Adapter load failed — using base model for {checkpoint_name}")

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────


def main() -> None:
    import torch

    from server.environment import FlightRebookingEnvironment
    from training.smoke_test import SYSTEM_PROMPT, generate_response, run_episode
    from training.space_runner import init_trackio

    parser = argparse.ArgumentParser(description="Evaluate checkpoints")
    parser.add_argument("--baseline", action="store_true", help="Evaluate base model")
    parser.add_argument("--sft", action="store_true", help="Evaluate SFT adapter")
    parser.add_argument("--phase1", action="store_true", help="Evaluate GRPO Phase 1")
    parser.add_argument("--phase2", action="store_true", help="Evaluate GRPO Phase 2")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per tier (default: 5)")
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=["easy", "medium", "hard"],
        help="Tiers to evaluate (default: all)",
    )
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda, mps, cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s — %(message)s")

    # Collect selected checkpoints
    selected: list[str] = []
    if args.baseline:
        selected.append("baseline")
    if args.sft:
        selected.append("sft")
    if args.phase1:
        selected.append("phase1")
    if args.phase2:
        selected.append("phase2")

    if not selected:
        parser.error("Select at least one checkpoint: --baseline, --sft, --phase1, --phase2")

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device     : {device}")
    print(f"Base model : {args.base_model}")
    print(f"Checkpoints: {', '.join(selected)}")
    print(f"Tiers      : {', '.join(args.tiers)}")
    print(f"Episodes   : {args.episodes} per tier")
    print()

    # ── Init trackio ────────────────────────────────────────
    init_trackio("eval", config={
        "checkpoints": selected,
        "tiers": args.tiers,
        "episodes_per_tier": args.episodes,
    })

    # ── Evaluate each checkpoint ────────────────────────────
    all_results: Dict[str, Dict[str, List[Dict]]] = {}

    for ckpt_name in selected:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {ckpt_name} — {CHECKPOINTS[ckpt_name]['description']}")
        print(f"{'=' * 60}")

        model, tokenizer = load_model_for_eval(
            ckpt_name,
            base_model=args.base_model,
            device=device,
        )

        # Build policy callable
        def make_policy(m, t, d):
            def policy(messages):
                return generate_response(m, t, messages, d)
            return policy

        policy = make_policy(model, tokenizer, device)

        ckpt_results: Dict[str, List[Dict]] = {}

        for tier in args.tiers:
            tier_results: List[Dict] = []
            for ep_idx in range(args.episodes):
                seed = ep_idx
                task_id = f"{tier}_{ep_idx:03d}"

                env = FlightRebookingEnvironment(debug=False)
                obs = env.reset(seed=seed, task_id=task_id)
                result = run_episode(env, obs, policy, SYSTEM_PROMPT)
                result["checkpoint"] = ckpt_name
                result["seed"] = seed
                tier_results.append(result)

            ckpt_results[tier] = tier_results

        all_results[ckpt_name] = ckpt_results

        # Free GPU memory between checkpoints
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary table ───────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Checkpoint':<12} {'Tier':<8} {'Episodes':<9} {'Mean Score':<12} {'Min':<8} {'Max':<8} {'Parse Clean'}")
    print(f"{'-' * 12} {'-' * 8} {'-' * 9} {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 11}")

    for ckpt_name in selected:
        for tier in args.tiers:
            results = all_results[ckpt_name][tier]
            scores = [r["grader_score"] for r in results]
            clean = sum(1 for r in results if r["parse_clean"])
            mean_score = sum(scores) / len(scores) if scores else 0.0
            min_score = min(scores) if scores else 0.0
            max_score = max(scores) if scores else 0.0
            print(
                f"{ckpt_name:<12} {tier:<8} {len(results):<9} "
                f"{mean_score:<12.4f} {min_score:<8.4f} {max_score:<8.4f} "
                f"{clean}/{len(results)}"
            )

    print(f"{'=' * 80}")

    # Log to trackio
    try:
        import trackio

        for ckpt_name in selected:
            for tier in args.tiers:
                results = all_results[ckpt_name][tier]
                scores = [r["grader_score"] for r in results]
                if scores:
                    trackio.log({
                        f"{ckpt_name}/{tier}/mean_score": sum(scores) / len(scores),
                        f"{ckpt_name}/{tier}/min_score": min(scores),
                        f"{ckpt_name}/{tier}/max_score": max(scores),
                    })
    except Exception:
        pass


if __name__ == "__main__":
    main()
