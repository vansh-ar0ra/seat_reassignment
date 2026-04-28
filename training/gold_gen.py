#!/usr/bin/env python3
"""Step 7 — Generate gold-standard trajectories using Gemini as teacher policy.

Runs Gemini over the 40 easy-tier SFT scenarios (data_sft/easy_000–easy_039),
filters by grader score and parse quality, and writes the results to
training/data/gold_trajectories.jsonl.

Usage:
    # Run on 2 scenarios (quick test):
    GEMINI_API_KEY=... python training/gold_gen.py --candidates 2

    # Full run on all 40 scenarios:
    GEMINI_API_KEY=... python training/gold_gen.py

    # Custom model and threshold:
    GEMINI_API_KEY=... python training/gold_gen.py --model gemini-2.5-flash --threshold 0.80
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Make repo root importable
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from google import genai

from client import FlightRebookingEnv
from training.smoke_test import SYSTEM_PROMPT, run_episode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-2.5-pro"
MIN_SCORE = 0.85
FALLBACK_MIN_SCORE = 0.80
TARGET_KEPT = 20
MAX_CANDIDATES = 40
MAX_TURNS_PER_EPISODE = 5
MAX_CONCURRENT = 5

DATA_SFT_DIR = Path(__file__).resolve().parent.parent / "data_sft"
OUTPUT_FILE = Path(__file__).resolve().parent / "data" / "gold_trajectories.jsonl"


# ---------------------------------------------------------------------------
# Gemini message conversion
# ---------------------------------------------------------------------------

def _to_gemini_contents(messages: List[Dict[str, str]]) -> List[dict]:
    """Convert OpenAI-style messages to Gemini contents format.

    System messages are stripped (handled via system_instruction).
    Maps 'assistant' role to 'model' for Gemini.
    """
    contents = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            continue  # handled separately via system_instruction
        gemini_role = "model" if role == "assistant" else "user"
        contents.append({
            "role": gemini_role,
            "parts": [{"text": msg["content"]}],
        })
    return contents


def _extract_system_prompt(messages: List[Dict[str, str]]) -> str:
    """Extract the system prompt from a message list."""
    for msg in messages:
        if msg["role"] == "system":
            return msg["content"]
    return ""


# ---------------------------------------------------------------------------
# Retry with backoff
# ---------------------------------------------------------------------------

def _retry_with_backoff(fn, *args, max_retries: int = 5, **kwargs):
    """Call fn with exponential backoff on rate-limit/server errors."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if any(code in err_str for code in ("429", "503", "RESOURCE_EXHAUSTED")):
                wait = 2 ** attempt
                print(f"  [RETRY] attempt {attempt + 1}/{max_retries}, "
                      f"waiting {wait}s: {err_str[:100]}")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Max retries ({max_retries}) exceeded")


# ---------------------------------------------------------------------------
# Gemini policy factory
# ---------------------------------------------------------------------------

def make_gemini_policy(client: genai.Client, model_name: str):
    """Return a PolicyCallable that calls Gemini."""

    def policy(messages: List[Dict[str, str]]) -> str:
        system_instruction = _extract_system_prompt(messages)
        contents = _to_gemini_contents(messages)

        response = _retry_with_backoff(
            client.models.generate_content,
            model=model_name,
            contents=contents,
            config={
                "system_instruction": system_instruction,
                "temperature": 0.7,
                "max_output_tokens": 8192,
            },
        )
        return response.text

    return policy


# ---------------------------------------------------------------------------
# Rate-limited wrapper
# ---------------------------------------------------------------------------

_rate_semaphore = threading.Semaphore(MAX_CONCURRENT)


def rate_limited_policy(inner_policy):
    """Wrap a policy callable with a concurrency semaphore."""

    def wrapper(messages: List[Dict[str, str]]) -> str:
        with _rate_semaphore:
            return inner_policy(messages)

    return wrapper


# ---------------------------------------------------------------------------
# Run single scenario
# ---------------------------------------------------------------------------

ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")


def run_single_scenario(
    task_id: str,
    client: genai.Client,
    model_name: str,
) -> Dict[str, Any]:
    """Run one episode for a given task_id and return the result dict."""
    with FlightRebookingEnv(base_url=ENV_URL).sync() as env_client:
        reset_result = env_client.reset(task_id=task_id, seed=0)
        obs = reset_result.observation

        policy = rate_limited_policy(make_gemini_policy(client, model_name))
        result = run_episode(env_client, obs, policy, SYSTEM_PROMPT)
        result["task_id"] = task_id
        result["seed"] = 0
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate gold-standard trajectories using Gemini"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=MAX_CANDIDATES,
        help=f"Number of SFT scenarios to run (default: {MAX_CANDIDATES})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MIN_SCORE,
        help=f"Minimum grader score to keep (default: {MIN_SCORE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_FILE),
        help=f"Output JSONL path (default: {OUTPUT_FILE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Max concurrent Gemini requests (default: {MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run scenarios sequentially (useful for debugging)",
    )
    args = parser.parse_args()

    # Update semaphore for custom worker count
    global _rate_semaphore
    _rate_semaphore = threading.Semaphore(args.workers)

    # Validate API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is required.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # List SFT scenario task_ids
    if not DATA_SFT_DIR.is_dir():
        print(f"ERROR: SFT data directory not found: {DATA_SFT_DIR}")
        sys.exit(1)

    task_ids = sorted([
        d.name for d in DATA_SFT_DIR.iterdir()
        if d.is_dir() and d.name.startswith("easy_")
    ])
    task_ids = task_ids[: args.candidates]

    print(f"Model      : {args.model}")
    print(f"Candidates : {len(task_ids)}")
    print(f"Threshold  : {args.threshold}")
    print(f"Output     : {args.output}")
    print(f"Workers    : {args.workers}")
    print(f"Sequential : {args.sequential}")
    print()

    # Run episodes
    results: List[Dict[str, Any]] = []

    if args.sequential:
        for i, task_id in enumerate(task_ids):
            print(f"\n[{i + 1}/{len(task_ids)}] Running {task_id}...")
            try:
                result = run_single_scenario(task_id, client, args.model)
                results.append(result)
                print(f"  Score: {result['grader_score']:.4f}  "
                      f"Parse clean: {result['parse_clean']}  "
                      f"Turns: {result['turns_used']}")
            except Exception as exc:
                print(f"  [ERROR] {task_id}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(
                    run_single_scenario, task_id, client, args.model
                ): task_id
                for task_id in task_ids
            }
            for i, future in enumerate(as_completed(future_to_task), 1):
                task_id = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"\n[{i}/{len(task_ids)}] {task_id}: "
                          f"score={result['grader_score']:.4f}  "
                          f"parse_clean={result['parse_clean']}  "
                          f"turns={result['turns_used']}")
                except Exception as exc:
                    print(f"\n[{i}/{len(task_ids)}] {task_id}: [ERROR] {exc}")

    # Filter
    threshold = args.threshold
    kept = [
        r for r in results
        if r["grader_score"] >= threshold
        and r["parse_clean"]
        and r["turns_used"] <= MAX_TURNS_PER_EPISODE
    ]

    print(f"\n{'=' * 60}")
    print(f"Filtering: {len(results)} total -> {len(kept)} pass "
          f"(threshold={threshold}, parse_clean=True, turns<={MAX_TURNS_PER_EPISODE})")

    # Fallback threshold
    if len(kept) < TARGET_KEPT and threshold > FALLBACK_MIN_SCORE:
        print(f"WARNING: Only {len(kept)} trajectories at >= {threshold}. "
              f"Lowering threshold to {FALLBACK_MIN_SCORE}")
        kept = [
            r for r in results
            if r["grader_score"] >= FALLBACK_MIN_SCORE
            and r["parse_clean"]
            and r["turns_used"] <= MAX_TURNS_PER_EPISODE
        ]
        print(f"  After lowering: {len(kept)} pass")

    if len(kept) < TARGET_KEPT:
        print(f"WARNING: Only {len(kept)} trajectories pass filter "
              f"(target: {TARGET_KEPT}). Dataset may be insufficient.")

    # Sort by score descending
    kept.sort(key=lambda r: r["grader_score"], reverse=True)

    # Write JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in kept:
            line = {
                "task_id": r["task_id"],
                "seed": r.get("seed", 0),
                "tier": "easy",
                "score": r["grader_score"],
                "turns_used": r["turns_used"],
                "messages": r["messages"],
            }
            f.write(json.dumps(line) + "\n")

    print(f"\nWrote {len(kept)} trajectories to {output_path}")

    # Summary table
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total scenarios run : {len(results)}")
    print(f"  Trajectories kept   : {len(kept)}")
    if kept:
        scores = [r["grader_score"] for r in kept]
        print(f"  Score range         : {min(scores):.4f} – {max(scores):.4f}")
        print(f"  Mean score          : {sum(scores) / len(scores):.4f}")
    if results:
        all_scores = [r["grader_score"] for r in results]
        failed = [r for r in results if not r["parse_clean"]]
        print(f"  All scores range    : {min(all_scores):.4f} – {max(all_scores):.4f}")
        print(f"  Parse failures      : {len(failed)}/{len(results)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
