"""TRL-compatible rollout function for the Flight Rebooking environment.

Exports ``rollout_func(prompts, trainer, **kwargs)`` which drives multi-turn
episodes against the remote env (via WebSocket client), builds token-level
completion masks (model tokens = 1, env/tool tokens = 0), and returns the dict
TRL's GRPOTrainer expects.

The function also returns five per-episode reward component keys that map 1-to-1
with the environment's grader sub-scores:
  coverage_reward, cabin_match_reward, group_integrity_reward,
  deadline_reward, ssr_integrity_reward

Each is selectable by an individual reward function in the training config.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from client import FlightRebookingEnv
from models import FlightRebookingAction

logger = logging.getLogger("flight_rebooking.rollout")

# ---------------------------------------------------------------------------
# Remote env configuration
# ---------------------------------------------------------------------------
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

# Module-level singleton, lazy-initialized (sync wrapper)
_env_client = None


def _get_env():
    """Return a persistent remote env client (lazy-initialized, sync)."""
    global _env_client
    if _env_client is None:
        _env_client = FlightRebookingEnv(base_url=ENV_URL).sync()
        _env_client.__enter__()
    return _env_client


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TURNS = 8
TEMPERATURE = 0.7
TOP_P = 0.9

# Task IDs cycled deterministically per prompt
TASK_IDS = ["easy", "medium", "hard"]

# Reward component names (must match grader terminal_breakdown keys)
REWARD_COMPONENTS = [
    "coverage_reward",
    "cabin_match_reward",
    "group_integrity_reward",
    "deadline_reward",
    "ssr_integrity_reward",
]

_BREAKDOWN_KEY_MAP = {
    "coverage_score": "coverage_reward",
    "cabin_match_score": "cabin_match_reward",
    "group_integrity_score": "group_integrity_reward",
    "deadline_score": "deadline_reward",
    "ssr_integrity_score": "ssr_integrity_reward",
}

# ---------------------------------------------------------------------------
# System prompt (shared with smoke_test.py and inference_ollama.py)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an airline rebooking agent. A flight was cancelled; rebook all passengers onto alternative flights. You place passengers into cabin buckets (not specific seats).

TOOLS (call one per turn inside <action> tags):
1. get_full_manifest() — returns all passengers: id, priority_tier(1=high,5=low), original_cabin, group_id, group_integrity(hard/soft/null), ssr_flags, downstream_deadline.
2. get_flight_inventory() — returns flights: flight_id, departure/arrival times, cabin availability, supports_ssr.
3. submit_plan(assignments) — submit COMPLETE plan: {"PAX-001":{"flight_id":"FL-201","cabin":"economy"},...}. ONE shot, no revisions.
4. finalize_plan() — lock in plan for final grading.

CONSTRAINTS (priority order):
- SSR COMPATIBILITY (HARD): passengers with ssr_flags (UM,WCHR,pet_cabin,pet_cargo) only go on flights supporting those flags.
- HARD GROUPS (HARD): all "hard" group members must be on the SAME flight.
- DEADLINES (HARD): arrival_time must be <= passenger's downstream_deadline.
- COVERAGE (0.35): rebook every passenger.
- CABIN MATCH (0.15): match original cabin; higher-priority passengers weighted more.
- SOFT GROUPS (0.15): keep "soft" groups together when possible.

WORKFLOW: get_full_manifest → get_flight_inventory → submit_plan → finalize_plan.
Think step-by-step, then output: <action>{"tool_name":"...","args":{...}}</action>"""

# Concise variant for gold trajectory generation (SFT training data).
# Gemini produces short, action-focused responses that a small model can imitate.
GOLD_SYSTEM_PROMPT = """\
You are an airline rebooking agent. A flight was cancelled; rebook all passengers onto alternative flights. You place passengers into cabin buckets (not specific seats).

TOOLS (call one per turn inside <action> tags):
1. get_full_manifest() — returns all passengers with priority_tier, original_cabin, group_id, group_integrity, ssr_flags, downstream_deadline.
2. get_flight_inventory() — returns flights with cabin_availability and supports_ssr.
3. submit_plan(assignments) — submit COMPLETE plan: {"PAX-001":{"flight_id":"FL-201","cabin":"economy"},...}. ONE shot, no revisions.
4. finalize_plan() — lock in plan for final grading.

CONSTRAINTS (priority order):
- SSR COMPATIBILITY (HARD): ssr_flags passengers only on flights supporting those flags.
- HARD GROUPS (HARD): all "hard" group members on the SAME flight.
- DEADLINES (HARD): arrival_time <= downstream_deadline.
- COVERAGE (0.35): rebook every passenger.
- CABIN MATCH (0.15): match original cabin; higher-priority passengers weighted more.
- SOFT GROUPS (0.15): keep "soft" groups together when possible.

WORKFLOW: get_full_manifest → get_flight_inventory → submit_plan → finalize_plan.

CRITICAL: Be maximally concise.
- For information-gathering steps (manifest/inventory): output ONLY the <action> tag, nothing else.
- For the planning step: write a brief 2-3 sentence rationale, then the <action> tag with your complete assignment JSON.
- For finalize: output ONLY the <action> tag.
- Never use verbose XML reasoning tags. No <observations>, <passenger_analysis>, <strategy>, <tradeoff_analysis>, <reconsideration>.
- Every response must contain exactly one <action>{"tool_name":"...","args":{...}}</action> block."""


# ╔═══════════════════════════════════════════════════════════════╗
# ║  XML / JSON Parsing                                          ║
# ╚═══════════════════════════════════════════════════════════════╝


def _extract_xml_tag(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def _repair_trailing_braces(text: str) -> str:
    d = text.count("{") - text.count("}")
    if d > 0:
        text += "}" * d
    d = text.count("[") - text.count("]")
    if d > 0:
        text += "]" * d
    return text


def _find_nested_json(text: str) -> Optional[dict]:
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
                    c = json.loads(text[start : i + 1])
                    if isinstance(c, dict) and "tool_name" in c:
                        c.setdefault("args", {})
                        return c
                except json.JSONDecodeError:
                    start = None
    if depth > 0 and start is not None:
        repaired = text[start:] + "}" * depth
        try:
            c = json.loads(repaired)
            if isinstance(c, dict) and "tool_name" in c:
                c.setdefault("args", {})
                return c
        except json.JSONDecodeError:
            pass
    return None


def parse_action(text: str) -> Optional[dict]:
    """Extract a tool-call dict from LLM output.  Returns None on failure."""
    text = text.strip()

    # Strategy 1: <action> tags
    action_inner = _extract_xml_tag(text, "action")
    if action_inner:
        inner = action_inner
        if "```" in inner:
            fence = re.search(r"```(?:json)?\s*(.*?)\s*```", inner, re.DOTALL)
            if fence:
                inner = fence.group(1).strip()
        inner = _repair_trailing_braces(inner)
        try:
            p = json.loads(inner)
            if isinstance(p, dict) and "tool_name" in p:
                p.setdefault("args", {})
                return p
        except json.JSONDecodeError:
            for m in re.finditer(r"\{[^{}]*\}", inner):
                try:
                    c = json.loads(m.group())
                    if "tool_name" in c:
                        c.setdefault("args", {})
                        return c
                except json.JSONDecodeError:
                    continue
            r = _find_nested_json(inner)
            if r:
                return r

    # Strategy 2: code fences
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()

    # Strategy 3: direct parse
    try:
        p = json.loads(text)
        if isinstance(p, dict) and "tool_name" in p:
            p.setdefault("args", {})
            return p
    except json.JSONDecodeError:
        pass

    # Strategy 4: flat JSON
    for m in re.finditer(r"\{[^{}]*\}", text):
        try:
            c = json.loads(m.group())
            if "tool_name" in c:
                c.setdefault("args", {})
                return c
        except json.JSONDecodeError:
            continue

    # Strategy 5: nested JSON
    return _find_nested_json(text)


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Message Formatting                                          ║
# ╚═══════════════════════════════════════════════════════════════╝


def _format_main_task() -> str:
    return (
        "Task: A flight has been cancelled. Rebook all passengers onto "
        "alternative flights, respecting constraints and priorities."
    )


def _format_state(obs: Any) -> str:
    parts = [
        f"=== Step {obs.step_count}/{obs.max_steps} | "
        f"Booked: {obs.passengers_booked}/{obs.passengers_total} | "
        f"Remaining: {obs.passengers_remaining} | "
        f"Plan submitted: {obs.plan_submitted} ==="
    ]
    if obs.booked_summary:
        parts.append("\nCurrent bookings:")
        for b in obs.booked_summary:
            parts.append(f"  {b['passenger_id']} -> {b['flight_id']} ({b['cabin']})")
    return "\n".join(parts)


def _format_instruction(step: int, plan_submitted: bool) -> str:
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


def _format_result(tool_result: Optional[dict], reward: Optional[float], reward_reason: Optional[str]) -> str:
    parts: list[str] = []
    if tool_result is not None:
        parts.append(f"Last tool result: {json.dumps(tool_result, indent=2)}")
    if reward is not None:
        parts.append(f"Reward: {reward:.4f} ({reward_reason or ''})")
    return "\n".join(parts) if parts else "Tool executed."


def _fallback_action(obs: Any) -> dict:
    if obs.step_count == 0:
        return {"tool_name": "get_full_manifest", "args": {}}
    if not obs.plan_submitted:
        return {"tool_name": "get_flight_inventory", "args": {}}
    return {"tool_name": "finalize_plan", "args": {}}


# ╔═══════════════════════════════════════════════════════════════╗
# ║  vLLM generation via generate_rollout_completions            ║
# ╚═══════════════════════════════════════════════════════════════╝


def _generate_via_trainer(
    trainer: Any,
    tokenizer: Any,
    messages: list[dict],
) -> tuple[list[int], list[int], list[float], str]:
    """Generate a single completion through the trainer's vLLM via TRL 0.29+.

    Returns (prompt_token_ids, completion_token_ids, per_token_logprobs, gen_text).
    """
    from trl.experimental.openenv import generate_rollout_completions

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    gen_out = generate_rollout_completions(trainer, [prompt_text])[0]
    prompt_ids = gen_out["prompt_ids"]
    completion_ids = gen_out["completion_ids"]
    logprobs = gen_out["logprobs"]
    gen_text = gen_out.get("text") or tokenizer.decode(
        completion_ids, skip_special_tokens=True
    )

    return prompt_ids, completion_ids, logprobs, gen_text


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Single-episode runner                                       ║
# ╚═══════════════════════════════════════════════════════════════╝


def _play_episode(
    trainer: Any,
    tokenizer: Any,
    task_id: str,
    seed: int,
) -> dict[str, Any]:
    """Run one multi-turn episode against the remote env.

    Returns a dict with:
        prompt_ids:       list[int]  — initial prompt token IDs
        completion_ids:   list[int]  — concatenated completion token IDs (all turns)
        logprobs:         list[float] — per-token logprobs for completion_ids
        env_mask:         list[int]  — 1 for model tokens, 0 for env tokens
        breakdown:        dict       — grader component scores
        grader_score:     float
    """
    env = _get_env()
    result = env.reset(task_id=task_id, seed=seed)
    obs = result.observation

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Accumulate tokens across turns
    all_completion_ids: list[int] = []
    all_logprobs: list[float] = []
    all_env_mask: list[int] = []

    # First prompt (will be the "prompt" portion — not in completion)
    first_user_msg = "\n\n".join([
        _format_main_task(),
        _format_state(obs),
        _format_instruction(obs.step_count, obs.plan_submitted),
    ])
    messages.append({"role": "user", "content": first_user_msg})

    # Tokenize the initial prompt (system + first user) — this is "prompt_ids"
    initial_prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(initial_prompt_text, add_special_tokens=False)

    consecutive_failures = 0
    history: list[dict] = []
    done = False

    for turn in range(1, MAX_TURNS + 1):
        if obs.done:
            break

        # Build messages for this turn (after turn 1, add env feedback)
        if turn > 1:
            last = history[-1]
            user_parts = [
                _format_result(last["tool_result"], last["reward"], last["reward_reason"]),
                _format_state(obs),
                _format_instruction(obs.step_count, obs.plan_submitted),
            ]
            user_msg = "\n\n".join(user_parts)
            messages.append({"role": "user", "content": user_msg})

            # Tokenize the env feedback (user message) — these tokens get mask=0
            full_text_before_gen = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            env_feedback_ids = tokenizer.encode(full_text_before_gen, add_special_tokens=False)
            env_token_count = len(env_feedback_ids) - len(prompt_ids) - len(all_completion_ids)
            if env_token_count > 0:
                all_completion_ids.extend(env_feedback_ids[-env_token_count:])
                all_logprobs.extend([0.0] * env_token_count)
                all_env_mask.extend([0] * env_token_count)

        # Generate via TRL's vLLM
        try:
            _, gen_ids, gen_logprobs, gen_text = _generate_via_trainer(
                trainer, tokenizer, messages
            )
        except Exception as exc:
            logger.warning("Generation error: %s", exc)
            gen_ids = []
            gen_logprobs = []
            gen_text = ""

        # Append model tokens — these get mask=1
        all_completion_ids.extend(gen_ids)
        all_logprobs.extend(gen_logprobs)
        all_env_mask.extend([1] * len(gen_ids))

        # Parse the action
        parsed = parse_action(gen_text)
        used_fallback = False

        if parsed is None:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                parsed = {"tool_name": "finalize_plan", "args": {}}
            else:
                parsed = _fallback_action(obs)
            used_fallback = True
        else:
            consecutive_failures = 0

        # Add assistant message to conversation
        messages.append({"role": "assistant", "content": gen_text})

        # Step the remote environment
        action = FlightRebookingAction(
            tool_name=parsed["tool_name"],
            args=parsed.get("args", {}),
        )
        step_result = env.step(action)
        obs = step_result.observation
        done = step_result.done

        history.append({
            "action": parsed,
            "tool_result": obs.tool_result,
            "reward": obs.reward,
            "reward_reason": obs.reward_reason,
        })

        if done or obs.done:
            break

    # Force finalize if not done
    if not done and not getattr(obs, "done", False):
        try:
            action = FlightRebookingAction(tool_name="finalize_plan", args={})
            step_result = env.step(action)
            obs = step_result.observation
        except RuntimeError as exc:
            logger.warning("finalize_plan after loop failed: %s", exc)

    # Extract terminal scores
    grader_score = 0.0
    breakdown: dict[str, float] = {}
    if obs.tool_result:
        grader_score = obs.tool_result.get("grader_score", 0.0)
        breakdown = obs.tool_result.get("terminal_breakdown", {})

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_mask": all_env_mask,
        "breakdown": breakdown,
        "grader_score": grader_score,
    }


# ╔═══════════════════════════════════════════════════════════════╗
# ║  rollout_func — the public TRL interface                     ║
# ╚═══════════════════════════════════════════════════════════════╝


def rollout_func(
    prompts: list[Any],
    trainer: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    """TRL GRPOTrainer-compatible rollout function.

    Signature: ``RolloutFunc = Callable[[list[str], GRPOTrainer], dict[str, Any]]``

    Args:
        prompts:  List of prompt strings (or message lists) from the dataset.
                  Each prompt maps to one episode.  The function is responsible
                  for returning ``num_generations`` completions per prompt.
        trainer:  The GRPOTrainer instance.  Used to access the tokenizer,
                  model (via vLLM), and ``num_generations`` config.

    Returns:
        Dict with required keys ``prompt_ids``, ``completion_ids``, ``logprobs``
        and optional ``env_mask`` (completion mask for multi-turn masking).
        Extra keys (the 5 reward components) are forwarded to reward functions.
    """
    tokenizer = trainer.processing_class
    num_generations = getattr(trainer.args, "num_generations", 1)

    print(f"[ROLLOUT] called with {len(prompts)} prompts, num_generations={num_generations}, "
          f"generation_batch_size={getattr(trainer.args, 'generation_batch_size', '?')}")

    # Accumulate per-sample results
    all_prompt_ids: list[list[int]] = []
    all_completion_ids: list[list[int]] = []
    all_logprobs: list[list[float]] = []
    all_env_mask: list[list[int]] = []

    # Reward components — one float per sample
    component_lists: dict[str, list[float]] = {k: [] for k in REWARD_COMPONENTS}

    for prompt_idx, prompt in enumerate(prompts):
        # Derive task_id and seed deterministically from the prompt index
        task_id = TASK_IDS[prompt_idx % len(TASK_IDS)]
        base_seed = prompt_idx * 1000
        # If the prompt is a dataset dict/row with task_id/seed, use those
        if isinstance(prompt, dict):
            task_id = prompt.get("task_id", task_id)
            base_seed = prompt.get("seed", base_seed)

        for gen_idx in range(num_generations):
            seed = base_seed + gen_idx

            episode = _play_episode(
                trainer=trainer,
                tokenizer=tokenizer,
                task_id=task_id,
                seed=seed,
            )

            all_prompt_ids.append(episode["prompt_ids"])
            all_completion_ids.append(episode["completion_ids"])
            all_logprobs.append(episode["logprobs"])
            all_env_mask.append(episode["env_mask"])

            # Map grader breakdown to reward component keys
            breakdown = episode.get("breakdown", {})
            for env_key, reward_key in _BREAKDOWN_KEY_MAP.items():
                component_lists[reward_key].append(
                    float(breakdown.get(env_key, 0.0))
                )

    result: dict[str, Any] = {
        # Required by TRL
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        # Completion mask: 1 = model token, 0 = env/tool token
        "env_mask": all_env_mask,
    }

    # Validate token IDs are in range (catches OOV before CUDA asserts)
    import torch as _torch
    _model_vocab = trainer.model.config.vocab_size
    for key in ("prompt_ids", "completion_ids"):
        val = result.get(key)
        if val is None:
            continue
        if isinstance(val, _torch.Tensor):
            max_id = val.max().item()
            min_id = val.min().item()
            shape = tuple(val.shape)
        else:
            flat = [t for seq in val for t in (seq.tolist() if hasattr(seq, "tolist") else seq)]
            if not flat:
                continue
            max_id = max(flat)
            min_id = min(flat)
            shape = (len(val), max(len(s) for s in val))
        print(f"[ROLLOUT CHECK] {key}: shape={shape}, min={min_id}, max={max_id}, "
              f"vocab={tokenizer.vocab_size}, model_vocab={_model_vocab}")
        assert 0 <= min_id, f"{key} has negative ID: {min_id}"
        assert max_id < _model_vocab, (
            f"{key} has OOV ID: {max_id} >= model.config.vocab_size={_model_vocab}"
        )

    # Reward component keys — forwarded to reward functions via extra_fields
    for key in REWARD_COMPONENTS:
        result[key] = component_lists[key]

    return result
