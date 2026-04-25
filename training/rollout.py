"""TRL-compatible rollout function for the Flight Rebooking environment.

Exports ``rollout_func(prompts, trainer, **kwargs)`` which drives multi-turn
episodes against the env, builds token-level completion masks (model tokens = 1,
env/tool tokens = 0), and returns the dict TRL's GRPOTrainer expects.

The function also returns five per-episode reward component keys that map 1-to-1
with the environment's grader sub-scores:
  coverage_reward, cabin_match_reward, group_integrity_reward,
  deadline_reward, ssr_integrity_reward

Each is selectable by an individual reward function in the training config.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from models import FlightRebookingAction
from server.environment import FlightRebookingEnvironment

logger = logging.getLogger("flight_rebooking.rollout")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TURNS = 8
MAX_NEW_TOKENS = 4096
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


def _format_result(tool_result: Optional[dict], reward: float, reward_reason: str) -> str:
    parts: list[str] = []
    if tool_result is not None:
        parts.append(f"Last tool result: {json.dumps(tool_result, indent=2)}")
    parts.append(f"Reward: {reward:.4f} ({reward_reason})")
    return "\n".join(parts) if parts else "Tool executed."


def _fallback_action(obs: Any) -> dict:
    if obs.step_count == 0:
        return {"tool_name": "get_full_manifest", "args": {}}
    if not obs.plan_submitted:
        return {"tool_name": "get_flight_inventory", "args": {}}
    return {"tool_name": "finalize_plan", "args": {}}


# ╔═══════════════════════════════════════════════════════════════╗
# ║  vLLM generation via the trainer                             ║
# ╚═══════════════════════════════════════════════════════════════╝


def _generate_via_trainer(
    trainer: Any,
    tokenizer: Any,
    messages: list[dict],
) -> tuple[list[int], list[int], list[float]]:
    """Generate a single completion through the trainer's vLLM client.

    Returns (prompt_token_ids, completion_token_ids, per_token_logprobs).
    """
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Use vLLM via the trainer's generation infrastructure
    if hasattr(trainer, "vllm_generation") and trainer.vllm_generation is not None:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_tokens=MAX_NEW_TOKENS,
            logprobs=1,
        )
        outputs = trainer.vllm_generation.generate(
            prompts=None,
            sampling_params=sampling_params,
            prompt_token_ids=[prompt_ids],
        )
        output = outputs[0].outputs[0]
        completion_ids = list(output.token_ids)
        logprobs = []
        for lp_dict in output.logprobs:
            # Each logprobs entry is a dict {token_id: Logprob}; pick the sampled one
            if lp_dict:
                logprobs.append(next(iter(lp_dict.values())).logprob)
            else:
                logprobs.append(0.0)
    else:
        # Fallback: use model.generate directly (for testing without vLLM)
        import torch

        model = trainer.model
        device = next(model.parameters()).device
        input_ids = torch.tensor([prompt_ids], device=device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        full_ids = output.sequences[0].tolist()
        completion_ids = full_ids[len(prompt_ids):]

        # Compute per-token log-probs from scores
        logprobs = []
        if output.scores:
            for step_idx, score_tensor in enumerate(output.scores):
                log_probs_dist = torch.log_softmax(score_tensor[0], dim=-1)
                if step_idx < len(completion_ids):
                    tok = completion_ids[step_idx]
                    logprobs.append(log_probs_dist[tok].item())
                else:
                    logprobs.append(0.0)
        # Pad if needed
        while len(logprobs) < len(completion_ids):
            logprobs.append(0.0)

    return prompt_ids, completion_ids, logprobs


# ╔═══════════════════════════════════════════════════════════════╗
# ║  Single-episode runner                                       ║
# ╚═══════════════════════════════════════════════════════════════╝


def _play_episode(
    trainer: Any,
    tokenizer: Any,
    task_id: str,
    seed: int,
) -> dict[str, Any]:
    """Run one multi-turn episode.  Mirrors the Step-2 smoke_test loop.

    Returns a dict with:
        prompt_ids:       list[int]  — initial prompt token IDs
        completion_ids:   list[int]  — concatenated completion token IDs (all turns)
        logprobs:         list[float] — per-token logprobs for completion_ids
        env_mask:         list[int]  — 1 for model tokens, 0 for env tokens
        breakdown:        dict       — grader component scores
        grader_score:     float
    """
    env = FlightRebookingEnvironment(debug=False)
    obs = env.reset(seed=seed, task_id=task_id)

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
            # We tokenize just the appended portion by computing the delta
            full_text_before_gen = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            env_feedback_ids = tokenizer.encode(full_text_before_gen, add_special_tokens=False)
            # The new env tokens are everything beyond the previous prompt+completion
            # We track this as the user message tokens
            env_token_count = len(env_feedback_ids) - len(prompt_ids) - len(all_completion_ids)
            if env_token_count > 0:
                # These tokens are env feedback — mask them out
                all_completion_ids.extend(env_feedback_ids[-env_token_count:])
                all_logprobs.extend([0.0] * env_token_count)
                all_env_mask.extend([0] * env_token_count)

        # Generate
        try:
            _, gen_ids, gen_logprobs = _generate_via_trainer(
                trainer, tokenizer, messages
            )
        except Exception as exc:
            logger.warning("Generation error: %s", exc)
            gen_ids = []
            gen_logprobs = []

        # Decode for parsing
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

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

        # Step the environment
        action = FlightRebookingAction(
            tool_name=parsed["tool_name"],
            args=parsed.get("args", {}),
        )
        obs = env.step(action)

        history.append({
            "action": parsed,
            "tool_result": obs.tool_result,
            "reward": obs.reward,
            "reward_reason": obs.reward_reason,
        })

        if obs.done:
            break

    # Force finalize if not done
    if not obs.done:
        obs = env.step(FlightRebookingAction(tool_name="finalize_plan", args={}))

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
        # If the prompt is a dataset dict/row with task_id/seed, use those
        if isinstance(prompt, dict):
            task_id = prompt.get("task_id", task_id)

        for gen_idx in range(num_generations):
            seed = prompt_idx * 1000 + gen_idx

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

    # Reward component keys — forwarded to reward functions via extra_fields
    for key in REWARD_COMPONENTS:
        result[key] = component_lists[key]

    return result
