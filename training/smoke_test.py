#!/usr/bin/env python3
"""
Step 2 — Standalone single-rollout smoke test (no TRL).

Runs one full episode against the Flight Rebooking environment using a base
Qwen model loaded via transformers (bfloat16).  Proves out the complete loop:
    generate → parse XML → dispatch tool → step env → collect reward → repeat

Usage:
    python training/smoke_test.py --seed 0
    python training/smoke_test.py --seed 0 --task medium
    python training/smoke_test.py --seed 0 --task hard --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Make repo root importable (works whether invoked from repo root or training/)
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import FlightRebookingAction
from server.environment import FlightRebookingEnvironment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TURNS = 8
MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.7
TOP_P = 0.9

REASONING_TAGS = [
    "observations",
    "passenger_analysis",
    "strategy",
    "tradeoff_analysis",
    "reconsideration",
]

# ---------------------------------------------------------------------------
# System prompt — identical to inference_ollama.py
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an airline operations agent handling flight cancellation rebooking. A scheduled \
flight has been cancelled and all passengers must be rebooked onto alternative flights \
to the same destination. You operate at the inventory level — placing passengers into \
available cabin buckets on flights, NOT assigning specific seats.

═══════════════════════════════════════════════════════════════════
TOOLS AVAILABLE (4 tools — call exactly one per turn)
═══════════════════════════════════════════════════════════════════

1. get_full_manifest()
   Returns ALL passenger details: ID, name, priority_tier (1=highest, 5=lowest), \
original_cabin, group_id, group_integrity (hard/soft/null), group_size, ssr_flags, \
downstream_deadline.

2. get_flight_inventory()
   Returns ALL available flights: flight_id, departure_time, arrival_time, \
cabin_availability (economy/premium_economy/business seat counts), supports_ssr.

3. submit_plan(assignments)
   Submit a COMPLETE rebooking plan. Format:
   {"PAX-001": {"flight_id": "FL-201", "cabin": "economy"}, ...}
   - Validated atomically per passenger. Each is accepted or rejected with a reason.
   - Returns per-passenger results, group integrity checks, and a score preview.
   - ONE submission allowed per episode — no revisions.

4. finalize_plan()
   Lock in your submitted plan and trigger final grading. Call after submit_plan.

═══════════════════════════════════════════════════════════════════
CONSTRAINT PRIORITY ORDER (highest to lowest)
═══════════════════════════════════════════════════════════════════

1. SSR COMPATIBILITY (HARD — severe penalty if violated)
   Passengers with special service requirements (UM, WCHR, pet_cabin, pet_cargo) \
can ONLY go on flights that support those exact SSR flags. A passenger needing \
"WCHR" CANNOT be placed on a flight that only supports "UM".

2. HARD GROUP INTEGRITY (HARD — severe penalty if violated)
   All members of a "hard" integrity group MUST be on the SAME flight. Splitting \
them across flights or leaving some unbooked triggers a hard penalty.

3. DOWNSTREAM DEADLINES (HARD — rejection if violated)
   If a passenger has a downstream_deadline (e.g., "14:30"), the flight's \
arrival_time must be at or before that time. The plan validator will REJECT \
any assignment that misses the deadline.

4. COVERAGE (weight: 0.35)
   Every passenger should be rebooked. Missing passengers directly hurt the score.

5. CABIN MATCHING (weight: 0.15)
   Place passengers in their original cabin class when possible. Priority-weighted: \
tier 1 passengers matter 1.5x, tier 5 passengers matter 0.6x.

6. SOFT GROUP INTEGRITY (weight: 0.15)
   Passengers in a "soft" group should be kept on the same flight when possible, \
but splitting is acceptable with a small penalty.

═══════════════════════════════════════════════════════════════════
REASONING PROTOCOL — Chain of Thought with XML Tags
═══════════════════════════════════════════════════════════════════

You MUST think step-by-step before producing any action. Use the following XML tags \
to structure your reasoning. The action JSON goes LAST, inside <action> tags.

PHASE 1 — After receiving manifest and flight data, ANALYZE:

<observations>
Summarize the key facts:
- Total passengers, cabin breakdown, constraint counts
- Total flights, capacity per cabin, SSR support per flight
- Total capacity vs demand per cabin class
- Identify bottlenecks (which cabins are tight?)
</observations>

<passenger_analysis>
For EACH constrained passenger (has SSR flags, deadline, or is in a group), analyze:
- Passenger ID, constraints, original cabin
- Which flights are eligible (meet SSR + deadline requirements)?
- If in a group: which flights can accommodate the ENTIRE group?
- Rank eligible flights by preference (cabin match > earlier departure)

For unconstrained passengers, summarize by cabin class:
- How many economy/premium_economy/business passengers with no constraints?
- Available capacity after constrained passengers are placed?
</passenger_analysis>

PHASE 2 — Plan construction:

<strategy>
Build the plan in this order:
1. SSR-constrained passengers → assign to SSR-compatible flights
2. Hard groups → find flights that fit ALL members together
3. Deadline-constrained passengers → flights arriving before deadline
4. Remaining passengers → fill by priority tier (highest first), matching cabin
Explicitly track remaining capacity as you assign each passenger/group.
</strategy>

<tradeoff_analysis>
If any conflicts exist (e.g., two constrained passengers competing for the same \
limited slot), analyze the tradeoff:
- What are the options?
- Which option gives a better overall score?
- Priority-weighted impact of each choice?
</tradeoff_analysis>

<reconsideration>
Before finalizing, review the full plan:
- Did every passenger get assigned?
- Are all hard constraints satisfied?
- Any cabin mismatches that could be fixed by swapping two passengers?
- Any capacity left unused that could improve placements?
If you find improvements, revise the assignments.
</reconsideration>

PHASE 3 — Output the action:

<action>
{"tool_name": "...", "args": {...}}
</action>

═══════════════════════════════════════════════════════════════════
WORKFLOW (you have 5 steps max)
═══════════════════════════════════════════════════════════════════

Step 1: Call get_full_manifest() to see all passengers and constraints.
Step 2: Call get_flight_inventory() to see all flights and capacity.
Step 3: Use full chain-of-thought reasoning, then call submit_plan(assignments) \
with a complete plan covering ALL passengers.
Step 4: Call finalize_plan() to lock in and get your final score.

IMPORTANT:
- You get ONE shot at submit_plan — no revisions. Reason carefully.
- Include ALL passengers — missing ones directly reduce your coverage score.
- Never violate SSR or hard group constraints — the penalty is severe (-0.15 each).
- Always output your action inside <action>...</action> tags.
- The JSON inside <action> tags must be valid JSON with no trailing commas.

For information-gathering steps (steps 1-2), your reasoning can be brief:

<observations>
Gathering passenger manifest data.
</observations>

<action>
{"tool_name": "get_full_manifest", "args": {}}
</action>
"""


# ╔═══════════════════════════════════════════════════════════════╗
# ║  XML / JSON Parsing (adapted from inference_ollama.py)       ║
# ╚═══════════════════════════════════════════════════════════════╝


def extract_xml_tag(text: str, tag: str) -> Optional[str]:
    """Extract content between <tag> and </tag>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def _repair_trailing_braces(text: str) -> str:
    """Append missing closing braces/brackets to truncated JSON."""
    open_b = text.count("{") - text.count("}")
    if open_b > 0:
        text += "}" * open_b
    open_sq = text.count("[") - text.count("]")
    if open_sq > 0:
        text += "]" * open_sq
    return text


def _find_nested_json(text: str) -> Optional[dict]:
    """Find a nested JSON object containing 'tool_name'."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    candidate = json.loads(text[start : i + 1])
                    if isinstance(candidate, dict) and "tool_name" in candidate:
                        candidate.setdefault("args", {})
                        return candidate
                except json.JSONDecodeError:
                    start = None
    # Repair truncated tail
    if depth > 0 and start is not None:
        repaired = text[start:] + "}" * depth
        try:
            candidate = json.loads(repaired)
            if isinstance(candidate, dict) and "tool_name" in candidate:
                candidate.setdefault("args", {})
                return candidate
        except json.JSONDecodeError:
            pass
    return None


def parse_llm_response(response_text: str) -> Optional[dict]:
    """Parse LLM response into a tool-call dict.  Returns None on failure.

    Extraction priority:
      1. <action>…</action> XML tags
      2. Markdown code fences
      3. Direct JSON parse
      4. First flat {...} with "tool_name"
      5. Nested JSON search
    """
    text = response_text.strip()

    # --- Strategy 1: <action> tags ---
    action_content = extract_xml_tag(text, "action")
    if action_content:
        inner = action_content
        if "```" in inner:
            fence = re.search(r"```(?:json)?\s*(.*?)\s*```", inner, re.DOTALL)
            if fence:
                inner = fence.group(1).strip()
        inner = _repair_trailing_braces(inner)
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict) and "tool_name" in parsed:
                parsed.setdefault("args", {})
                return parsed
        except json.JSONDecodeError:
            # Flat blocks inside action tag
            for m in re.finditer(r"\{[^{}]*\}", inner):
                try:
                    c = json.loads(m.group())
                    if "tool_name" in c:
                        c.setdefault("args", {})
                        return c
                except json.JSONDecodeError:
                    continue
            result = _find_nested_json(inner)
            if result:
                return result

    # --- Strategy 2: code fences ---
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

    # --- Strategy 3: direct parse ---
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "tool_name" in parsed:
            parsed.setdefault("args", {})
            return parsed
    except json.JSONDecodeError:
        pass

    # --- Strategy 4: flat JSON blocks ---
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            c = json.loads(m.group())
            if "tool_name" in c:
                c.setdefault("args", {})
                return c
        except json.JSONDecodeError:
            continue

    # --- Strategy 5: nested JSON ---
    return _find_nested_json(text)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Message Formatting                                          ║
# ╚═══════════════════════════════════════════════════════════════╝


def format_main_task() -> str:
    return (
        "Task: A flight has been cancelled. Rebook all passengers onto "
        "alternative flights, respecting constraints and priorities."
    )


def format_state(obs) -> str:
    parts = [
        f"=== Step {obs.step_count}/{obs.max_steps} | "
        f"Booked: {obs.passengers_booked}/{obs.passengers_total} | "
        f"Remaining: {obs.passengers_remaining} | "
        f"Plan submitted: {obs.plan_submitted} ==="
    ]
    if obs.booked_summary:
        parts.append("\nCurrent bookings:")
        for b in obs.booked_summary:
            parts.append(
                f"  {b['passenger_id']} -> {b['flight_id']} ({b['cabin']})"
            )
    return "\n".join(parts)


def format_instruction(step: int, plan_submitted: bool) -> str:
    if step == 0:
        return (
            "This is the start of the episode. Call get_full_manifest() to see "
            "all passengers. Wrap your action in <action>...</action> tags."
        )
    if step == 1:
        return (
            "You have the passenger manifest. Now call get_flight_inventory() to "
            "see all flights. Wrap your action in <action>...</action> tags."
        )
    if not plan_submitted and step >= 2:
        return (
            "You now have all the data. Reason through constraints carefully using "
            "the XML thinking tags (<observations>, <passenger_analysis>, <strategy>, "
            "<tradeoff_analysis>, <reconsideration>), then output your complete "
            "rebooking plan inside <action>...</action> tags. "
            "Include ALL passengers. This is your ONE shot — no revisions."
        )
    if plan_submitted:
        return (
            "Your plan has been submitted. Call finalize_plan() to lock in your "
            "score. Wrap your action in <action>...</action> tags."
        )
    return "Choose your next action. Wrap it in <action>...</action> tags."


def format_result(
    tool_result: Optional[dict], reward: float, reward_reason: str
) -> str:
    parts: list[str] = []
    if tool_result is not None:
        parts.append(f"Last tool result: {json.dumps(tool_result, indent=2)}")
    parts.append(f"Reward: {reward:.4f} ({reward_reason})")
    return "\n".join(parts) if parts else "Tool executed."


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Fallback Action                                             ║
# ╚═══════════════════════════════════════════════════════════════╝


def fallback_action(obs) -> dict:
    """Deterministic fallback when the model produces unparseable output."""
    if obs.step_count == 0:
        return {"tool_name": "get_full_manifest", "args": {}}
    if not obs.plan_submitted:
        return {"tool_name": "get_flight_inventory", "args": {}}
    return {"tool_name": "finalize_plan", "args": {}}


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Model Generation                                            ║
# ╚═══════════════════════════════════════════════════════════════╝


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: List[dict],
    device: str,
) -> str:
    """Run a single forward pass through the model and decode new tokens."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Episode Runner                                              ║
# ╚═══════════════════════════════════════════════════════════════╝


def run_episode(
    model: Any,
    tokenizer: Any,
    task_id: str,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    """Run a single episode and return a result dict."""

    env = FlightRebookingEnvironment(debug=False)
    obs = env.reset(seed=seed, task_id=task_id)

    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    history: List[dict] = []
    rewards: List[float] = []
    turn_logs: List[dict] = []
    raw_outputs: List[str] = []
    parse_failures = 0
    consecutive_failures = 0

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"Episode: task={task_id}  seed={seed}")
    print(
        f"Passengers: {obs.passengers_total}  |  "
        f"Max env steps: {obs.max_steps}  |  Hard cap: {MAX_TURNS} turns"
    )
    print(sep)

    for turn in range(1, MAX_TURNS + 1):
        if obs.done:
            break

        # ---- build user message ----
        if turn == 1:
            user_content = "\n\n".join(
                [
                    format_main_task(),
                    format_state(obs),
                    format_instruction(obs.step_count, obs.plan_submitted),
                ]
            )
            messages.append({"role": "user", "content": user_content})
        else:
            last = history[-1]
            user_parts = [
                format_result(
                    last["tool_result"], last["reward"], last["reward_reason"]
                ),
                format_state(obs),
                format_instruction(obs.step_count, obs.plan_submitted),
            ]
            messages.append({"role": "user", "content": "\n\n".join(user_parts)})

        # ---- generate ----
        print(f"\n--- Turn {turn}/{MAX_TURNS} ---")
        t0 = time.time()
        try:
            raw_response = generate_response(model, tokenizer, messages, device)
        except Exception as exc:
            raw_response = ""
            print(f"  [GEN ERROR] {exc}")
        gen_time = time.time() - t0

        raw_outputs.append(raw_response)
        print(f"  Generated {len(raw_response)} chars in {gen_time:.1f}s")

        # ---- log reasoning tags ----
        for tag in REASONING_TAGS:
            content = extract_xml_tag(raw_response, tag)
            if content:
                preview = content[:150].replace("\n", " ")
                print(f"  <{tag}> {preview}...")

        action_xml = extract_xml_tag(raw_response, "action")
        if action_xml:
            print(f"  <action> {action_xml[:300]}")
        else:
            print(f"  [raw preview] {raw_response[:300]}")

        # ---- parse ----
        parsed = parse_llm_response(raw_response)
        used_fallback = False

        if parsed is None:
            parse_failures += 1
            consecutive_failures += 1
            print(f"  [PARSE FAIL #{parse_failures}]")
            print(f"  Raw (first 500 chars): {raw_response[:500]}")

            if consecutive_failures >= 3:
                print(
                    "  [ABORT] 3 consecutive parse failures — forcing finalize"
                )
                parsed = {"tool_name": "finalize_plan", "args": {}}
            else:
                parsed = fallback_action(obs)
                print(f"  [FALLBACK] {parsed['tool_name']}")
            used_fallback = True
        else:
            consecutive_failures = 0
            print(f"  [PARSED] tool={parsed['tool_name']}")

        # ---- append assistant turn to message history ----
        messages.append({"role": "assistant", "content": raw_response})

        # ---- step the environment ----
        action = FlightRebookingAction(
            tool_name=parsed["tool_name"],
            args=parsed.get("args", {}),
        )
        obs = env.step(action)

        reward = obs.reward
        rewards.append(reward)

        print(
            f"  Reward: {reward:+.4f}  |  "
            f"Cumulative: {obs.cumulative_reward:.4f}  |  "
            f"Booked: {obs.passengers_booked}/{obs.passengers_total}  |  "
            f"Done: {obs.done}"
        )

        turn_logs.append(
            {
                "turn": turn,
                "tool_name": parsed["tool_name"],
                "args_keys": list(parsed.get("args", {}).keys()),
                "used_fallback": used_fallback,
                "reward": round(reward, 6),
                "cumulative_reward": round(obs.cumulative_reward, 6),
                "booked": obs.passengers_booked,
                "done": obs.done,
                "raw_len": len(raw_response),
                "gen_time_s": round(gen_time, 2),
            }
        )

        history.append(
            {
                "action": parsed,
                "tool_result": obs.tool_result,
                "reward": reward,
                "reward_reason": obs.reward_reason,
                "raw_response": raw_response,
            }
        )

        if obs.done:
            break

    # ---- hard-cap: force finalize if still not done ----
    if not obs.done:
        print(f"\n  [HARD CAP] Reached {MAX_TURNS} turns — forcing finalize_plan")
        obs = env.step(
            FlightRebookingAction(tool_name="finalize_plan", args={})
        )
        rewards.append(obs.reward)
        turn_logs.append(
            {
                "turn": len(turn_logs) + 1,
                "tool_name": "finalize_plan",
                "args_keys": [],
                "used_fallback": True,
                "reward": round(obs.reward, 6),
                "cumulative_reward": round(obs.cumulative_reward, 6),
                "booked": obs.passengers_booked,
                "done": obs.done,
                "raw_len": 0,
                "gen_time_s": 0.0,
            }
        )

    # ---- extract terminal scores ----
    grader_score = 0.0
    breakdown: dict = {}
    if obs.tool_result:
        grader_score = obs.tool_result.get("grader_score", 0.0)
        breakdown = obs.tool_result.get("terminal_breakdown", {})

    result = {
        "task_id": task_id,
        "seed": seed,
        "grader_score": round(grader_score, 6),
        "breakdown": {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in breakdown.items()
        },
        "cumulative_reward": round(obs.cumulative_reward, 6),
        "turns_used": len(turn_logs),
        "parse_failures": parse_failures,
        "rewards": [round(r, 6) for r in rewards],
        "turn_logs": turn_logs,
        "raw_outputs": raw_outputs,
    }

    # ---- print summary ----
    print(f"\n{sep}")
    print(f"EPISODE RESULT  task={task_id}  seed={seed}")
    print(f"  Grader score    : {grader_score:.4f}")
    print(f"  Cumulative rew  : {obs.cumulative_reward:.4f}")
    print(f"  Turns used      : {len(turn_logs)}/{MAX_TURNS}")
    print(f"  Parse failures  : {parse_failures}")
    if breakdown:
        print("  Breakdown:")
        for k, v in breakdown.items():
            label = k.replace("_", " ").title()
            if isinstance(v, float):
                print(f"    {label:25s}: {v:.4f}")
            else:
                print(f"    {label:25s}: {v}")
    print(sep)

    return result


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Main                                                        ║
# ╚═══════════════════════════════════════════════════════════════╝


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test: single rollout with base model (no TRL)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="easy",
        choices=["easy", "medium", "hard", "all"],
        help="Difficulty tier or 'all' to run all three (default: easy)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, mps, cpu (auto-detect if omitted)",
    )
    args = parser.parse_args()

    # ---- device selection ----
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Task   : {args.task}")
    print(f"Seed   : {args.seed}")

    # ---- load model ----
    print(f"\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model (bfloat16)...")
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        )
        model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}\n")

    # ---- run episodes ----
    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    all_results: List[Dict[str, Any]] = []

    for task_id in tasks:
        result = run_episode(model, tokenizer, task_id, args.seed, device)
        all_results.append(result)

    # ---- final summary ----
    print("\n" + "#" * 60)
    print("FINAL REWARD DICT")
    print("#" * 60)
    for r in all_results:
        summary = {
            "task_id": r["task_id"],
            "seed": r["seed"],
            "grader_score": r["grader_score"],
            "breakdown": r["breakdown"],
            "cumulative_reward": r["cumulative_reward"],
            "turns_used": r["turns_used"],
            "parse_failures": r["parse_failures"],
        }
        print(json.dumps(summary, indent=2))
    print("#" * 60)


if __name__ == "__main__":
    main()
