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
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "openai>=1.0.0", "--quiet"],
            timeout=120,
        )
    except Exception:
        pass
    from openai import OpenAI

try:
    from client import FlightRebookingEnv
    from models import FlightRebookingAction
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import FlightRebookingEnv
    from models import FlightRebookingAction

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"
IMAGE_NAME = os.getenv("IMAGE_NAME")
ENV_URL = os.getenv("ENV_URL") or os.getenv("SERVER_URL") or "http://localhost:8000"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BENCHMARK = "flight_rebooking"
TEMPERATURE = 0.3
MAX_TOKENS = 1000

# ---------------------------------------------------------------------------
# System prompt (constant across all difficulty tiers)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an airline operations agent. A scheduled flight has been cancelled and all passengers must be rebooked onto alternative flights to the same destination. You operate at the inventory level — placing passengers into available cabin buckets on flights, NOT assigning specific seats.

YOUR GOAL:
Produce a rebooking plan that gets every passenger to their destination while respecting constraints in this priority order:

1. HARD CONSTRAINTS (must not violate):
   - SSR compatibility: passengers with special service requirements (UM, WCHR, pet_cabin, pet_cargo) can only go on flights that support those SSRs.
   - Hard group integrity: passengers in a "hard" group must all be on the same flight.
   - Downstream deadlines: if a passenger has a connection deadline, their new flight must arrive by that time.

2. COVERAGE: every passenger should be rebooked onto some flight.

3. CABIN MATCHING: place passengers in their original cabin class (economy, premium_economy, business) when possible.

4. PRIORITY TIERS: higher-priority passengers (tier 1 is highest, tier 5 is lowest) should get better outcomes when trade-offs are needed.

5. SOFT GROUP INTEGRITY: passengers in a "soft" group should be kept together when possible, but splitting is acceptable.

6. FALLBACK ORDER: if original cabin is unavailable, try split-cabin on same flight before splitting across flights.

TOOLS AVAILABLE:
Each turn you must call exactly one tool.

1. list_passengers()
   - Returns a summary of all passengers: ID, priority tier, group ID, and flags for SSR/deadline.
   - Call this first to plan your approach.

2. get_passenger_details(passenger_id)
   - Returns full details: original cabin, SSR flags, group membership, deadline, priority.
   - Use before booking a passenger whose constraints you need to check.

3. list_alternative_flights()
   - Returns all available flights with per-cabin seat counts, times, and SSR support.
   - Seat counts update after bookings. Call again to refresh availability.

4. get_flight_details(flight_id)
   - Returns details for one specific flight including current availability.

5. book_passenger(passenger_id, flight_id, cabin)
   - Books one passenger onto a flight in the specified cabin (economy|premium_economy|business).
   - Will be rejected if: no seats, SSR mismatch, deadline violation, or already booked.
   - For hard group members, prefer book_group instead.

6. book_group(group_id, flight_id, cabin_assignments)
   - Books an entire group onto one flight atomically. All succeed or all fail.
   - cabin_assignments is a dict mapping each passenger_id to their cabin.

7. finalize_plan()
   - Call this when you are done. Triggers final scoring. Unbooked passengers count as failures.

ACTION FORMAT:
Respond with ONLY a raw JSON object. No reasoning, no markdown, no extra text.
Examples:
{"tool_name": "list_passengers", "args": {}}
{"tool_name": "get_passenger_details", "args": {"passenger_id": "PAX-001"}}
{"tool_name": "list_alternative_flights", "args": {}}
{"tool_name": "get_flight_details", "args": {"flight_id": "FL-201"}}
{"tool_name": "book_passenger", "args": {"passenger_id": "PAX-001", "flight_id": "FL-201", "cabin": "business"}}
{"tool_name": "book_group", "args": {"group_id": "GRP-001", "flight_id": "FL-201", "cabin_assignments": {"PAX-002": "economy", "PAX-003": "economy"}}}
{"tool_name": "finalize_plan", "args": {}}

STRATEGY:
1. Start with list_passengers and list_alternative_flights to survey the situation.
2. Identify constrained passengers: those with SSR flags, deadlines, or hard group membership.
3. Book the most constrained passengers first (hard groups, SSR+deadline combos).
4. Then book remaining passengers in priority-tier order, matching original cabin.
5. Use book_group for groups (especially hard groups) to keep them together atomically.
6. After all passengers are booked, call finalize_plan.
7. If a booking fails, check why and try an alternative flight or cabin.

IMPORTANT:
- Always call list_passengers first to understand the problem.
- Never violate hard constraints — the penalty is severe.
- Book hard groups with book_group, not individual book_passenger calls.
- Call finalize_plan when done — unbooked passengers hurt your score."""

# ---------------------------------------------------------------------------
# Task definitions: (task_name, task_id, max_steps)
# ---------------------------------------------------------------------------
TASKS = [
    ("task_easy",   "easy",   30),
    ("task_medium", "medium", 60),
    ("task_hard",   "hard",   90),
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
        f"Remaining: {obs.passengers_remaining} ==="
    ]

    if obs.booked_summary:
        parts.append("\nCurrent bookings:")
        for b in obs.booked_summary:
            parts.append(f"  {b['passenger_id']} -> {b['flight_id']} ({b['cabin']})")

    if obs.flights_snapshot:
        parts.append("\nFlight availability:")
        for fl in obs.flights_snapshot:
            avail = ", ".join(
                f"{c}={n}" for c, n in fl["cabin_availability"].items() if n > 0
            )
            parts.append(
                f"  {fl['flight_id']} dep={fl['departure_time']} arr={fl['arrival_time']} "
                f"SSR={fl['supports_ssr']} [{avail}]"
            )

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
# Action parsing
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str) -> Optional[dict]:
    """Parse LLM response into a tool call dict. Returns None on failure."""
    text = response_text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        parsed = json.loads(text)
        if "tool_name" in parsed and "args" in parsed:
            return parsed
        return None
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*"tool_name"[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_agent_action(
    client: OpenAI, obs, conversation_history: list, task_id: str
) -> Optional[dict]:
    """Call the LLM to get the next action."""
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

        recent_history = conversation_history[-6:]
        for i, item in enumerate(recent_history):
            messages.append(
                {"role": "assistant", "content": json.dumps(item["action"])}
            )

            user_parts = [format_result(item)]
            if i == len(recent_history) - 1:
                user_parts.append(format_state(obs))
                user_parts.append(format_instruction())

            messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        return parse_llm_response(response_text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Fallback action
# ---------------------------------------------------------------------------

def fallback_action(obs) -> dict:
    """Simple fallback: list passengers if nothing booked yet, else list flights."""
    if obs.passengers_booked == 0:
        return {"tool_name": "list_passengers", "args": {}}
    return {"tool_name": "list_alternative_flights", "args": {}}


# ---------------------------------------------------------------------------
# Mandatory logging
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
        env = FlightRebookingEnv(base_url=ENV_URL)
        result = await env.reset(task_id=task_id)
        obs = result.observation

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_dict = get_agent_action(client, obs, conversation_history, task_id)

            if action_dict is None:
                action_dict = fallback_action(obs)

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

    except Exception:
        pass

    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main — loop over all tasks
# ---------------------------------------------------------------------------

async def main() -> None:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    except Exception as exc:
        print(f"[ERROR] Failed to create OpenAI client: {exc}", flush=True)
        return
    for task_name, task_id, max_steps in TASKS:
        try:
            await run_task(task_name, task_id, max_steps, client)
        except Exception as exc:
            print(f"[ERROR] task={task_name} failed: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
