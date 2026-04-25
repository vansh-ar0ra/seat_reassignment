"""
Inference script for Flight Rebooking using a local Ollama Gemma model.

Prerequisites:
  1. Install Ollama:  brew install ollama
  2. Start the server: ollama serve
  3. Pull the model:   ollama pull gemma4:e4b
  4. Start the env:    uv run server
  5. Run this script:  python inference_ollama.py

Environment variables (all optional):
  OLLAMA_BASE_URL  – Ollama API base (default: http://localhost:11434/v1)
  OLLAMA_MODEL     – model tag to use   (default: gemma4:e4b)
  ENV_URL          – environment server  (default: http://localhost:8000)
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from typing import List, Optional

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
MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
ENV_URL = os.getenv("ENV_URL") or os.getenv("SERVER_URL") or "http://localhost:8000"

BENCHMARK = "flight_rebooking"
TEMPERATURE = 0.3
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# System prompt (identical to the main inference.py – constant across tiers)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an airline operations agent. A scheduled flight has been cancelled and \
all passengers must be rebooked onto alternative flights to the same destination. \
You operate at the inventory level — placing passengers into available cabin \
buckets on flights, NOT assigning specific seats.

YOUR GOAL:
Produce a rebooking plan that gets every passenger to their destination while \
respecting constraints in this priority order:

1. HARD CONSTRAINTS (must not violate):
   - SSR compatibility: passengers with special service requirements (UM, WCHR, \
pet_cabin, pet_cargo) can only go on flights that support those SSRs.
   - Hard group integrity: passengers in a "hard" group must all be on the same flight.
   - Downstream deadlines: if a passenger has a connection deadline, their new \
flight must arrive by that time.

2. COVERAGE: every passenger should be rebooked onto some flight.

3. CABIN MATCHING: place passengers in their original cabin class \
(economy, premium_economy, business) when possible.

4. PRIORITY TIERS: higher-priority passengers (tier 1 is highest, tier 5 is \
lowest) should get better outcomes when trade-offs are needed.

5. SOFT GROUP INTEGRITY: passengers in a "soft" group should be kept together \
when possible, but splitting is acceptable.

TOOLS AVAILABLE:
Each turn you must call exactly one tool.

1. get_full_manifest()
   - Returns ALL passenger details in one call: ID, name, priority tier, original cabin, \
group info, SSR flags, deadlines.
   - Call this first to understand all passengers and their constraints.

2. get_flight_inventory()
   - Returns ALL available flights with per-cabin seat counts, departure/arrival times, \
and SSR support.
   - Call this to understand available capacity and constraints.

3. submit_plan(assignments)
   - Submit a complete rebooking plan mapping each passenger to a flight and cabin.
   - assignments is a dict: {"PAX-001": {"flight_id": "FL-201", "cabin": "business"}, ...}
   - The plan is validated atomically. Each passenger is either accepted or rejected with a reason.
   - Returns per-passenger results, group integrity checks, and a score preview.
   - Only ONE submission allowed per episode.

4. finalize_plan()
   - Lock in your submitted plan and trigger final grading.
   - Unbooked passengers (rejected or missing from plan) count as failures.
   - Call this after reviewing your submit_plan results.

ACTION FORMAT:
Respond with ONLY a raw JSON object. No reasoning, no markdown, no extra text.
Examples:
{"tool_name": "get_full_manifest", "args": {}}
{"tool_name": "get_flight_inventory", "args": {}}
{"tool_name": "submit_plan", "args": {"assignments": {"PAX-001": {"flight_id": "FL-201", "cabin": "business"}, "PAX-002": {"flight_id": "FL-201", "cabin": "economy"}}}}
{"tool_name": "finalize_plan", "args": {}}

STRATEGY:
1. Call get_full_manifest() to see all passengers, their constraints, groups, SSR needs, and deadlines.
2. Call get_flight_inventory() to see all flights, their capacity, times, and SSR support.
3. Reason about constraints:
   - Match SSR passengers to compatible flights only.
   - Place hard group members on the same flight.
   - Respect downstream deadlines.
   - Match original cabins when possible.
   - Handle capacity limits across all flights.
4. Submit a complete plan with submit_plan() covering ALL passengers.
5. Review the results (accepted/rejected counts, constraint violations).
6. Call finalize_plan() to lock in and get your final score.

IMPORTANT:
- You have very few steps. Be efficient — gather info, plan carefully, submit, finalize.
- Never violate hard constraints — the penalty is severe.
- Include ALL passengers in your plan to maximize coverage score.
- Call finalize_plan when done — unbooked passengers hurt your score."""

# ---------------------------------------------------------------------------
# Task definitions: (task_name, task_id, max_steps)
# ---------------------------------------------------------------------------
TASKS = [
    # ("task_easy",   "easy",   5),
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


def format_instruction() -> str:
    return "Choose your next tool call. Respond with ONLY a JSON object."


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
# Action parsing — more robust for smaller models
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str) -> Optional[dict]:
    """Parse LLM response into a tool call dict. Returns None on failure.

    Smaller open-source models sometimes emit extra commentary around the JSON,
    so this parser tries several extraction strategies.
    """
    text = response_text.strip()

    # Strip markdown code fences
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Strategy 1: direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "tool_name" in parsed:
            parsed.setdefault("args", {})
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2: find the first {...} block containing "tool_name"
    for match in re.finditer(r"\{[^{}]*\}", text):
        try:
            candidate = json.loads(match.group())
            if "tool_name" in candidate:
                candidate.setdefault("args", {})
                return candidate
        except json.JSONDecodeError:
            continue

    # Strategy 3: find nested JSON (for book_group with cabin_assignments dict)
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

    return None


# ---------------------------------------------------------------------------
# LLM call via Ollama (OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

def get_agent_action(
    client: OpenAI, obs, conversation_history: list, task_id: str
) -> Optional[dict]:
    """Call the local Ollama model to get the next action."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if not conversation_history:
        user_content = "\n\n".join([
            format_main_task(task_id),
            format_state(obs),
            format_instruction(),
        ])
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": format_main_task(task_id)})

        # Keep a sliding window of recent history to stay within context limits.
        # Gemma models have smaller context than large proprietary models.
        recent_history = conversation_history[-4:]
        for i, item in enumerate(recent_history):
            messages.append(
                {"role": "assistant", "content": json.dumps(item["action"])}
            )

            user_parts = [format_result(item)]
            if i == len(recent_history) - 1:
                user_parts.append(format_state(obs))
                user_parts.append(format_instruction())

            messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    # Retry up to 3 times if the model returns unparseable output
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = (completion.choices[0].message.content or "").strip()
            print(f"  [LLM raw] {response_text[:200]}", flush=True)

            parsed = parse_llm_response(response_text)
            if parsed is not None:
                return parsed

            # On parse failure, append the bad response and ask again
            if attempt < 2:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": (
                        "Your response was not valid JSON. "
                        "Respond with ONLY a JSON object like: "
                        '{"tool_name": "...", "args": {...}}'
                    ),
                })
                print(f"  [RETRY] attempt {attempt + 2}/3 — response was not valid JSON", flush=True)

        except Exception as exc:
            print(f"  [LLM error] {exc}", flush=True)
            break

    return None


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

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_dict = get_agent_action(client, obs, conversation_history, task_id)

            if action_dict is None:
                action_dict = fallback_action(obs)
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

            conversation_history.append({
                "action": action_dict,
                "result": obs.tool_result,
                "reward": reward,
                "reward_reason": obs.reward_reason,
            })

            if done:
                if obs.tool_result and "grader_score" in obs.tool_result:
                    score = obs.tool_result["grader_score"]
                break

        success = score >= 0.5
        score = min(max(score, 0.0), 1.0)

    except Exception as exc:
        print(f"  [ERROR] {exc}", flush=True)

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
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
    print("Flight Rebooking — Ollama / Gemma Inference", flush=True)
    print(f"  Ollama endpoint : {OLLAMA_BASE_URL}", flush=True)
    print(f"  Model           : {MODEL_NAME}", flush=True)
    print(f"  Environment     : {ENV_URL}", flush=True)
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
