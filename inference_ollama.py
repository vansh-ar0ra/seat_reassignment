"""
Inference script for Flight Rebooking using a local Ollama Gemma model.

Prerequisites:
  1. Install Ollama:  brew install ollama
  2. Start the server: ollama serve
  3. Pull the model:   ollama pull qwen3:4b-instruct-fp16
  4. Start the env:    uv run server
  5. Run this script:  python inference_ollama.py

Environment variables (all optional):
  OLLAMA_BASE_URL  – Ollama API base (default: http://localhost:11434/v1)
  OLLAMA_MODEL     – model tag to use   (default: qwen3:4b-instruct-fp16)
  ENV_URL          – environment server  (default: http://localhost:8000)
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "openai>=1.0.0", "--quiet"],
        timeout=120,
    )
    from openai import OpenAI

try:
    from client import FlightRebookingEnv
    from models import FlightRebookingAction
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import FlightRebookingEnv
    from models import FlightRebookingAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-fp16")
ENV_URL = os.getenv("ENV_URL") or os.getenv("SERVER_URL") or "http://localhost:8000"

BENCHMARK = "flight_rebooking"
TEMPERATURE = 0.3
MAX_TOKENS = 8192  # higher limit for chain-of-thought reasoning
LOGS_DIR = os.getenv("INFERENCE_LOGS_DIR", "inference_logs")

REASONING_TAGS = [
    "observations", "passenger_analysis", "strategy",
    "tradeoff_analysis", "reconsideration",
]


# ---------------------------------------------------------------------------
# Structured inference logger
# ---------------------------------------------------------------------------

class InferenceLogger:
    """Persists full inference runs to disk in a structured directory.

    Output per run:
        inference_logs/<task_id>_<timestamp>/
            meta.json          — run metadata (model, task, config, timing)
            steps/
                step_001.json  — full record for step 1
                step_002.json  — full record for step 2
                ...
            summary.json       — episode outcome (score, rewards, steps)
            transcript.md      — human-readable transcript of the full run
    """

    def __init__(self, base_dir: str = LOGS_DIR):
        self._base = Path(base_dir)
        self._run_dir: Optional[Path] = None
        self._steps_dir: Optional[Path] = None
        self._step_records: List[dict] = []
        self._start_time: Optional[datetime] = None

    def start_run(self, task_id: str, task_name: str) -> str:
        """Create the run directory. Returns the path."""
        self._start_time = datetime.now()
        ts = self._start_time.strftime("%Y-%m-%dT%H-%M-%S")
        self._run_dir = self._base / f"{task_id}_{ts}"
        self._steps_dir = self._run_dir / "steps"
        self._steps_dir.mkdir(parents=True, exist_ok=True)
        self._step_records = []

        meta = {
            "task_id": task_id,
            "task_name": task_name,
            "model": MODEL_NAME,
            "ollama_base_url": OLLAMA_BASE_URL,
            "env_url": ENV_URL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "started_at": self._start_time.isoformat(),
        }
        self._write_json(self._run_dir / "meta.json", meta)
        return str(self._run_dir)

    def log_step(
        self,
        step: int,
        raw_llm_response: Optional[str],
        parsed_action: Optional[dict],
        used_fallback: bool,
        tool_result: Optional[dict],
        reward: float,
        reward_reason: str,
        cumulative_reward: float,
        observation_state: dict,
        done: bool,
        error: Optional[str],
        retry_count: int = 0,
        llm_attempts: Optional[List[dict]] = None,
        messages_sent: Optional[List[dict]] = None,
    ) -> None:
        """Write a full step record to disk.

        ``llm_attempts`` is a list of per-attempt records from
        ``get_agent_action``.  Each entry contains the raw LLM response,
        whether it parsed successfully, and any parse/LLM error.  This
        captures ALL retry attempts — including those that failed to
        produce a valid action.
        """
        if not self._run_dir:
            return

        # Extract reasoning tags from the *successful* raw response
        reasoning = {}
        if raw_llm_response:
            for tag in REASONING_TAGS:
                content = extract_xml_tag(raw_llm_response, tag)
                if content:
                    reasoning[tag] = content
            action_tag = extract_xml_tag(raw_llm_response, "action")
            if action_tag:
                reasoning["action_tag"] = action_tag

        # Build enriched attempt records with extracted reasoning
        enriched_attempts = []
        for att in (llm_attempts or []):
            enriched: Dict[str, Any] = dict(att)
            att_raw = att.get("raw_response")
            if att_raw:
                att_reasoning = {}
                for tag in REASONING_TAGS:
                    content = extract_xml_tag(att_raw, tag)
                    if content:
                        att_reasoning[tag] = content
                att_action = extract_xml_tag(att_raw, "action")
                if att_action:
                    att_reasoning["action_tag"] = att_action
                enriched["reasoning_tags"] = att_reasoning
            enriched_attempts.append(enriched)

        record: Dict[str, Any] = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "llm_response": {
                "raw": raw_llm_response,
                "reasoning_tags": reasoning,
                "char_count": len(raw_llm_response) if raw_llm_response else 0,
            },
            "parsed_action": parsed_action,
            "used_fallback": used_fallback,
            "retry_count": retry_count,
            "llm_attempts": enriched_attempts,
            "tool_result": tool_result,
            "reward": round(reward, 6),
            "reward_reason": reward_reason,
            "cumulative_reward": round(cumulative_reward, 6),
            "observation_state": observation_state,
            "done": done,
            "error": error,
        }

        if messages_sent:
            record["messages_sent"] = messages_sent

        self._step_records.append(record)
        filename = f"step_{step:03d}.json"
        self._write_json(self._steps_dir / filename, record)

    def finish_run(
        self,
        score: float,
        success: bool,
        steps_taken: int,
        rewards: List[float],
        error: Optional[str] = None,
    ) -> None:
        """Write the summary and transcript."""
        if not self._run_dir:
            return

        end_time = datetime.now()
        duration = (
            (end_time - self._start_time).total_seconds()
            if self._start_time else 0.0
        )

        summary = {
            "score": round(score, 6),
            "success": success,
            "steps_taken": steps_taken,
            "rewards": [round(r, 6) for r in rewards],
            "cumulative_reward": round(sum(rewards), 6),
            "duration_seconds": round(duration, 2),
            "started_at": self._start_time.isoformat() if self._start_time else None,
            "finished_at": end_time.isoformat(),
            "error": error,
            "tool_sequence": [
                r["parsed_action"].get("tool_name", "?")
                for r in self._step_records
                if r.get("parsed_action")
            ],
            "fallback_count": sum(
                1 for r in self._step_records if r.get("used_fallback")
            ),
            "total_retries": sum(
                r.get("retry_count", 0) for r in self._step_records
            ),
            "total_llm_calls": sum(
                len(r.get("llm_attempts", [])) for r in self._step_records
            ),
            "failed_parse_count": sum(
                1 for r in self._step_records
                for att in r.get("llm_attempts", [])
                if not att.get("parsed_ok", False)
            ),
        }
        self._write_json(self._run_dir / "summary.json", summary)
        self._write_transcript()

    def _write_transcript(self) -> None:
        """Write a human-readable markdown transcript."""
        if not self._run_dir:
            return

        lines: List[str] = []
        lines.append("# Inference Run Transcript")
        lines.append("")

        # Meta
        meta_path = self._run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            lines.append(f"**Task:** {meta.get('task_name', '?')} ({meta.get('task_id', '?')})")
            lines.append(f"**Model:** {meta.get('model', '?')}")
            lines.append(f"**Started:** {meta.get('started_at', '?')}")
            lines.append("")

        # Summary
        summary_path = self._run_dir / "summary.json"
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            lines.append("## Result")
            lines.append("")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| Score | {s.get('score', '?')} |")
            lines.append(f"| Success | {s.get('success', '?')} |")
            lines.append(f"| Steps | {s.get('steps_taken', '?')} |")
            lines.append(f"| Duration | {s.get('duration_seconds', '?')}s |")
            lines.append(f"| Tool sequence | {' → '.join(s.get('tool_sequence', []))} |")
            lines.append(f"| Fallbacks | {s.get('fallback_count', 0)} |")
            lines.append(f"| Retries | {s.get('total_retries', 0)} |")
            lines.append("")

        # Per-step details
        lines.append("## Steps")
        lines.append("")

        for record in self._step_records:
            step = record["step"]
            action = record.get("parsed_action") or {}
            tool = action.get("tool_name", "?")
            reward = record.get("reward", 0)
            done = record.get("done", False)
            fallback = record.get("used_fallback", False)

            retry_count = record.get("retry_count", 0)
            llm_attempts = record.get("llm_attempts", [])

            step_label = f"### Step {step}: `{tool}`"
            if retry_count > 0:
                step_label += f" ({retry_count} retries)"
            if fallback:
                step_label += " — FALLBACK"
            lines.append(step_label)
            lines.append("")

            if fallback:
                lines.append(
                    "> **Used fallback action** — LLM failed to produce "
                    f"valid output after {len(llm_attempts)} attempt(s)"
                )
                lines.append("")

            # --- Render ALL LLM attempts ---
            for att in llm_attempts:
                att_num = att.get("attempt", "?")
                att_ok = att.get("parsed_ok", False)
                att_raw = att.get("raw_response")
                att_reasoning = att.get("reasoning_tags", {})
                att_parse_err = att.get("parse_error")
                att_llm_err = att.get("llm_error")
                att_chars = att.get("char_count", 0)

                if len(llm_attempts) > 1 or not att_ok:
                    status = "PARSED" if att_ok else "FAILED"
                    lines.append(
                        f"#### Attempt {att_num}/{len(llm_attempts)} "
                        f"— **{status}** ({att_chars} chars)"
                    )
                    lines.append("")

                if att_llm_err:
                    lines.append(f"> **LLM error:** `{att_llm_err}`")
                    lines.append("")
                    continue

                # Reasoning tags from this attempt
                for tag in REASONING_TAGS:
                    if tag in att_reasoning:
                        lines.append(f"**<{tag}>**")
                        lines.append("")
                        lines.append("```")
                        lines.append(att_reasoning[tag])
                        lines.append("```")
                        lines.append("")

                # Action tag from this attempt
                if "action_tag" in att_reasoning:
                    lines.append("**<action>**")
                    lines.append("")
                    lines.append("```json")
                    lines.append(att_reasoning["action_tag"])
                    lines.append("```")
                    lines.append("")

                if att_parse_err:
                    lines.append(f"> **Parse error:** {att_parse_err}")
                    lines.append("")

                # If no reasoning tags were extracted, show raw response
                if not att_reasoning and att_raw:
                    raw_preview = att_raw[:3000]
                    if len(att_raw) > 3000:
                        raw_preview += "\n... (truncated)"
                    lines.append("**Raw LLM response:**")
                    lines.append("")
                    lines.append("```")
                    lines.append(raw_preview)
                    lines.append("```")
                    lines.append("")

            # If no attempts were recorded but we have a direct raw response
            if not llm_attempts:
                reasoning = record.get("llm_response", {}).get("reasoning_tags", {})
                for tag in REASONING_TAGS:
                    if tag in reasoning:
                        lines.append(f"**<{tag}>**")
                        lines.append("")
                        lines.append("```")
                        lines.append(reasoning[tag])
                        lines.append("```")
                        lines.append("")
                if "action_tag" in reasoning:
                    lines.append("**<action>**")
                    lines.append("")
                    lines.append("```json")
                    lines.append(reasoning["action_tag"])
                    lines.append("```")
                    lines.append("")
                elif action:
                    lines.append("**Parsed action:**")
                    lines.append("")
                    lines.append("```json")
                    lines.append(json.dumps(action, indent=2))
                    lines.append("```")
                    lines.append("")

            # Tool result (truncated for readability)
            tool_result = record.get("tool_result")
            if tool_result:
                result_str = json.dumps(tool_result, indent=2)
                if len(result_str) > 2000:
                    result_str = result_str[:2000] + "\n... (truncated)"
                lines.append("**Tool result:**")
                lines.append("")
                lines.append("```json")
                lines.append(result_str)
                lines.append("```")
                lines.append("")

            # Reward
            lines.append(
                f"**Reward:** {reward:+.4f} | "
                f"**Cumulative:** {record.get('cumulative_reward', 0):.4f} | "
                f"**Done:** {done}"
            )
            if record.get("error"):
                lines.append(f"**Error:** {record['error']}")
            lines.append("")
            lines.append("---")
            lines.append("")

        with open(self._run_dir / "transcript.md", "w") as f:
            f.write("\n".join(lines))

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# System prompt — chain-of-thought reasoning with XML tags
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

# ---------------------------------------------------------------------------
# Task definitions: (task_name, task_id, max_steps)
# ---------------------------------------------------------------------------
TASKS = [
    ("task_easy",   "easy",   5),
    ("task_medium", "medium", 5),
    ("task_hard",   "hard",   5),
]

# ---------------------------------------------------------------------------
# State and prompt formatting
# ---------------------------------------------------------------------------

def format_main_task(task_id: str) -> str:
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
            parts.append(f"  {b['passenger_id']} -> {b['flight_id']} ({b['cabin']})")

    return "\n".join(parts)


def format_instruction(step: int, plan_submitted: bool) -> str:
    """Return a phase-specific instruction hint."""
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


def format_result(item: dict) -> str:
    parts = []
    if item.get("result") is not None:
        parts.append(f"Last tool result: {json.dumps(item['result'], indent=2)}")
    if item.get("reward") is not None:
        parts.append(
            f"Reward: {item['reward']:.2f} ({item.get('reward_reason', 'unknown')})"
        )
    if not parts:
        return "Tool executed."
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Action parsing — extracts from <action> tags or falls back to JSON search
# ---------------------------------------------------------------------------

def extract_xml_tag(text: str, tag: str) -> Optional[str]:
    """Extract content between <tag> and </tag>."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _repair_trailing_braces(text: str) -> str:
    """Append missing closing braces/brackets to truncated JSON.

    Small LLMs often close the </action> XML tag before finishing
    deeply-nested JSON, leaving the string with unmatched '{'.
    This counts unmatched openers and appends the corresponding
    closers so that ``json.loads`` can succeed.
    """
    open_braces = text.count("{")
    close_braces = text.count("}")
    if open_braces > close_braces:
        text += "}" * (open_braces - close_braces)
    open_brackets = text.count("[")
    close_brackets = text.count("]")
    if open_brackets > close_brackets:
        text += "]" * (open_brackets - close_brackets)
    return text


def parse_llm_response(response_text: str) -> Optional[dict]:
    """Parse LLM response into a tool call dict. Returns None on failure.

    Extraction priority:
    1. <action>...</action> XML tags (preferred)
    2. Direct JSON parse
    3. JSON inside markdown code fences
    4. First {...} block containing "tool_name"
    5. Nested JSON search for complex args
    """
    text = response_text.strip()

    # Strategy 1: Extract from <action> tags
    action_content = extract_xml_tag(text, "action")
    if action_content:
        # Strip markdown fences if present inside action tags
        inner = action_content
        if "```" in inner:
            fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", inner, re.DOTALL)
            if fence_match:
                inner = fence_match.group(1).strip()
        # Auto-repair: append missing closing braces (common LLM truncation)
        inner = _repair_trailing_braces(inner)
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict) and "tool_name" in parsed:
                parsed.setdefault("args", {})
                return parsed
        except json.JSONDecodeError:
            # Try finding JSON within the action tag content
            for match in re.finditer(r"\{[^{}]*\}", inner):
                try:
                    candidate = json.loads(match.group())
                    if "tool_name" in candidate:
                        candidate.setdefault("args", {})
                        return candidate
                except json.JSONDecodeError:
                    continue
            # Try nested JSON within action tag
            result = _find_nested_json(inner)
            if result:
                return result

    # Strategy 2: Strip markdown code fences from full text
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Strategy 3: Direct parse of full text
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "tool_name" in parsed:
            parsed.setdefault("args", {})
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 4: Find the first {...} block containing "tool_name"
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            candidate = json.loads(match.group())
            if "tool_name" in candidate:
                candidate.setdefault("args", {})
                return candidate
        except json.JSONDecodeError:
            continue

    # Strategy 5: Nested JSON (for submit_plan with complex assignments dict)
    result = _find_nested_json(text)
    if result:
        return result

    return None


def _find_nested_json(text: str) -> Optional[dict]:
    """Find a nested JSON object containing 'tool_name' in text.

    If the text ends with unmatched braces (truncated JSON), the
    function appends missing closers and retries.
    """
    brace_depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start is not None:
                try:
                    candidate = json.loads(text[start : i + 1])
                    if isinstance(candidate, dict) and "tool_name" in candidate:
                        candidate.setdefault("args", {})
                        return candidate
                except json.JSONDecodeError:
                    start = None

    # If we ended with unmatched braces, try repairing the tail
    if brace_depth > 0 and start is not None:
        repaired = text[start:] + "}" * brace_depth
        try:
            candidate = json.loads(repaired)
            if isinstance(candidate, dict) and "tool_name" in candidate:
                candidate.setdefault("args", {})
                return candidate
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# LLM call via Ollama (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

def get_agent_action(
    client: OpenAI, obs, conversation_history: list, task_id: str
) -> Optional[dict]:
    """Call the local Ollama model to get the next action.

    Returns a parsed action dict on success (with ``_raw_response`` and
    ``_attempts`` keys attached), or ``None`` on total failure.  On failure
    the list of all attempted LLM responses is still available via the
    ``_attempts`` key — callers should retrieve it via
    ``action_dict.pop("_attempts", [])`` or, on ``None`` return, from the
    module-level ``last_failed_attempts`` list.
    """
    global last_failed_attempts
    last_failed_attempts = []
    attempts: List[Dict[str, Any]] = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if not conversation_history:
        user_content = "\n\n".join([
            format_main_task(task_id),
            format_state(obs),
            format_instruction(obs.step_count, obs.plan_submitted),
        ])
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": format_main_task(task_id)})

        # Include full conversation history — the model needs all data
        # to build a complete plan (manifest + inventory from prior steps)
        for i, item in enumerate(conversation_history):
            # Include the raw LLM response if available, otherwise the action JSON
            assistant_content = item.get("raw_response")
            if not assistant_content:
                assistant_content = json.dumps(item["action"])
            messages.append({"role": "assistant", "content": assistant_content})

            user_parts = [format_result(item)]
            if i == len(conversation_history) - 1:
                user_parts.append(format_state(obs))
                user_parts.append(
                    format_instruction(obs.step_count, obs.plan_submitted)
                )

            messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    # Retry up to 3 times if the model returns unparseable output
    for attempt in range(3):
        attempt_record: Dict[str, Any] = {
            "attempt": attempt + 1,
            "timestamp": datetime.now().isoformat(),
            "raw_response": None,
            "parsed_ok": False,
            "parse_error": None,
            "llm_error": None,
        }
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = (completion.choices[0].message.content or "").strip()
            attempt_record["raw_response"] = response_text
            attempt_record["char_count"] = len(response_text)

            # Log reasoning tags if present (truncated for readability)
            for tag in REASONING_TAGS:
                content = extract_xml_tag(response_text, tag)
                if content:
                    preview = content[:120].replace("\n", " ")
                    print(f"  [<{tag}>] {preview}...", flush=True)

            action_text = extract_xml_tag(response_text, "action")
            if action_text:
                print(f"  [<action>] {action_text[:200]}", flush=True)
            else:
                print(f"  [LLM raw] {response_text[:200]}", flush=True)

            parsed = parse_llm_response(response_text)
            if parsed is not None:
                attempt_record["parsed_ok"] = True
                attempts.append(attempt_record)

                # Store the raw response and attempts for the caller
                parsed["_raw_response"] = response_text
                parsed["_attempts"] = attempts
                return parsed

            # Parse failed — record why
            attempt_record["parse_error"] = "Could not extract valid JSON action from response"
            attempts.append(attempt_record)

            # On parse failure, append the bad response and ask again
            if attempt < 2:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": (
                        "I could not parse your response. Please output your action "
                        "inside <action>...</action> tags as valid JSON. Example:\n\n"
                        "<action>\n"
                        '{"tool_name": "get_full_manifest", "args": {}}\n'
                        "</action>"
                    ),
                })
                print(f"  [RETRY] attempt {attempt + 2}/3 — could not parse action", flush=True)

        except Exception as exc:
            attempt_record["llm_error"] = str(exc)
            attempts.append(attempt_record)
            print(f"  [LLM error] {exc}", flush=True)
            break

    # Total failure — store attempts so the caller can still log them
    last_failed_attempts = attempts
    return None


# Module-level storage for failed attempts when get_agent_action returns None.
last_failed_attempts: List[Dict[str, Any]] = []


# ---------------------------------------------------------------------------
# Fallback action
# ---------------------------------------------------------------------------

def fallback_action(obs) -> dict:
    """Simple fallback when the model fails to produce a valid action."""
    if obs.step_count == 0:
        return {"tool_name": "get_full_manifest", "args": {}}
    if not obs.plan_submitted:
        return {"tool_name": "get_flight_inventory", "args": {}}
    return {"tool_name": "finalize_plan", "args": {}}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Per-task episode runner
# ---------------------------------------------------------------------------

async def run_task(
    task_name: str,
    task_id: str,
    max_steps: int,
    client: OpenAI,
) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    conversation_history: list = []
    run_error: Optional[str] = None

    logger = InferenceLogger()
    run_dir = logger.start_run(task_id=task_id, task_name=task_name)
    print(f"  [LOG] Writing to {run_dir}", flush=True)

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    env = None
    try:
        # Use generous timeouts — local Ollama on CPU/MPS can take minutes
        # per response, and the default websockets ping timeout (20s) will
        # kill the connection while we wait for the LLM.
        env = FlightRebookingEnv(
            base_url=ENV_URL,
            message_timeout_s=600.0,   # 10 min per message round-trip
            connect_timeout_s=30.0,
        )
        # Monkey-patch the internal ws_connect call to disable ping
        # timeout so the websockets library doesn't drop the connection
        # while we're waiting for Ollama.
        _orig_connect = env.connect

        async def _connect_no_ping_timeout():
            """Connect with ping timeout disabled."""
            from websockets.asyncio.client import connect as _ws_connect
            if env._ws is not None:
                return env
            env._ws = await _ws_connect(
                env._ws_url,
                open_timeout=env._connect_timeout,
                max_size=env._max_message_size,
                ping_timeout=None,      # disable pong-wait timeout
                ping_interval=30,       # still send pings to keep NAT alive
                close_timeout=60,
            )
            return env

        env.connect = _connect_no_ping_timeout
        result = await env.reset(task_id=task_id)
        obs = result.observation
        cumulative_reward = 0.0

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_dict = get_agent_action(client, obs, conversation_history, task_id)

            raw_response = None
            used_fallback = False
            llm_attempts: List[Dict[str, Any]] = []
            if action_dict is not None:
                raw_response = action_dict.pop("_raw_response", None)
                llm_attempts = action_dict.pop("_attempts", [])

            if action_dict is None:
                # Capture the failed attempts before they're lost
                llm_attempts = last_failed_attempts.copy()
                action_dict = fallback_action(obs)
                used_fallback = True
                print(f"  [FALLBACK] Using fallback action: {action_dict['tool_name']}", flush=True)

            action_summary = (
                f"{action_dict['tool_name']}({json.dumps(action_dict['args'])})"
            )

            result = await env.step(
                FlightRebookingAction(
                    tool_name=action_dict["tool_name"],
                    args=action_dict["args"],
                )
            )
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            cumulative_reward += reward
            steps_taken = step

            error = (
                obs.tool_result.get("message")
                if obs.tool_result and obs.tool_result.get("status") == "error"
                else None
            )
            log_step(
                step=step, action=action_summary,
                reward=reward, done=done, error=error,
            )

            # Persist step to disk
            obs_state = {
                "step_count": obs.step_count,
                "max_steps": obs.max_steps,
                "passengers_total": obs.passengers_total,
                "passengers_booked": obs.passengers_booked,
                "passengers_remaining": obs.passengers_remaining,
                "plan_submitted": obs.plan_submitted,
                "booked_summary": obs.booked_summary,
            }
            logger.log_step(
                step=step,
                raw_llm_response=raw_response,
                parsed_action=action_dict,
                used_fallback=used_fallback,
                tool_result=obs.tool_result,
                reward=reward,
                reward_reason=obs.reward_reason,
                cumulative_reward=cumulative_reward,
                observation_state=obs_state,
                done=done,
                error=error,
                retry_count=max(0, len(llm_attempts) - 1),
                llm_attempts=llm_attempts,
            )

            conversation_history.append({
                "action": action_dict,
                "result": obs.tool_result,
                "reward": reward,
                "reward_reason": obs.reward_reason,
                "raw_response": raw_response,
            })

            if done:
                if obs.tool_result and "grader_score" in obs.tool_result:
                    score = obs.tool_result["grader_score"]
                break

        success = score >= 0.5
        score = min(max(score, 0.0), 1.0)

    except Exception as exc:
        run_error = str(exc)
        print(f"  [ERROR] {exc}", flush=True)

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        logger.finish_run(
            score=score,
            success=success,
            steps_taken=steps_taken,
            rewards=rewards,
            error=run_error,
        )
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Ollama health check
# ---------------------------------------------------------------------------

def check_ollama() -> bool:
    """Verify Ollama is reachable and the model is available."""
    import urllib.request
    import urllib.error

    # Check server is running
    base = OLLAMA_BASE_URL.replace("/v1", "")
    try:
        urllib.request.urlopen(f"{base}/api/tags", timeout=5)
    except (urllib.error.URLError, OSError):
        print(
            f"[ERROR] Cannot reach Ollama at {base}\n"
            "  Make sure Ollama is running:  ollama serve",
            flush=True,
        )
        return False

    # Check model is pulled
    try:
        req = urllib.request.Request(
            f"{base}/api/show",
            data=json.dumps({"name": MODEL_NAME}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(
                f"[ERROR] Model '{MODEL_NAME}' not found in Ollama.\n"
                f"  Pull it first:  ollama pull {MODEL_NAME}",
                flush=True,
            )
            return False
        # Other HTTP errors — model may still work
    except (urllib.error.URLError, OSError):
        pass

    print(f"[OK] Ollama reachable, model={MODEL_NAME}", flush=True)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60, flush=True)
    print("Flight Rebooking — Ollama / Gemma Inference (CoT mode)", flush=True)
    print(f"  Ollama endpoint : {OLLAMA_BASE_URL}", flush=True)
    print(f"  Model           : {MODEL_NAME}", flush=True)
    print(f"  Environment     : {ENV_URL}", flush=True)
    print(f"  Max tokens      : {MAX_TOKENS}", flush=True)
    print("=" * 60, flush=True)

    if not check_ollama():
        return

    # Ollama's OpenAI-compatible endpoint doesn't need a real API key,
    # but the openai library requires a non-empty string.
    # Set a long timeout since 31B models on local hardware can be slow.
    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        timeout=600.0,  # 10 min — gemma4:e4b can be slow on CPU/MPS
    )

    for task_name, task_id, max_steps in TASKS:
        print(f"\n{'─' * 40}", flush=True)
        print(f"Running {task_name} (task_id={task_id}, max_steps={max_steps})", flush=True)
        print(f"{'─' * 40}", flush=True)
        try:
            await run_task(task_name, task_id, max_steps, client)
        except Exception as exc:
            print(f"[ERROR] task={task_name} failed: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
