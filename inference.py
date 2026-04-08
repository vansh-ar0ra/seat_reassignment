import asyncio
import json
import os
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the project root (one level up from this file's directory)
load_dotenv(Path(__file__).parent / ".env")

from client import SeatReassignmentEnv
from models import SeatReassignmentAction

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"#"Qwen/Qwen2.5-72B-Instruct"
IMAGE_NAME = os.getenv("IMAGE_NAME")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BENCHMARK = "seat_reassignment"
TEMPERATURE = 0.3
MAX_TOKENS = 1000

# ---------------------------------------------------------------------------
# System prompt (single unified prompt for all tasks)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an airline operations agent. An aircraft swap has occurred — passengers from Aircraft-1 (AC-1) must be reassigned to Aircraft-2 (AC-2). The two aircraft have the same cabin classes but different seating layouts.

YOUR GOAL:
Reassign every passenger from AC-1 to AC-2, satisfying constraints in this priority order:
1. CABIN CLASS (mandatory) — business passengers must go to business seats, economy passengers must go to economy seats.
2. WINDOW PREFERENCE (if applicable) — if a passenger has paid_window=True, assign them to a window seat on AC-2.
3. LEGROOM PREFERENCE (if applicable) — if a passenger has paid_legroom=True, assign them to a seat with extra legroom on AC-2.

Only enforce the preferences that exist for a given passenger. If a passenger has no paid preferences, cabin class alone matters.

TOOLS AVAILABLE:
You have up to three tools depending on the task. Each turn you must call exactly one tool.

1. get_passenger_details(seat_id)
   - Fetches the passenger info for an AC-1 seat
   - Returns: passenger_id, name, cabin, and any applicable preference fields (e.g. paid_window, paid_legroom)
   - Use this to learn a passenger's preferences before assigning them

2. assign_seat(passenger_id, target_seat_id)
   - Moves a passenger from AC-1 to a specific AC-2 seat
   - The passenger must still be on AC-1; the AC-2 seat must be empty
   - Returns: confirmation including cabin_match and any preference satisfaction fields

3. swap_seats(passenger_id_1, passenger_id_2)  [if available]
   - Swaps two passengers who are BOTH already on AC-2
   - Use this to correct earlier misplacements

ACTION FORMAT:
Respond with ONLY a raw JSON object and absolutely no other text.
Do not include any reasoning, conversational text, or explanations.
Do NOT use markdown code blocks (e.g., ```json or ```). Respond ONLY with the JSON itself.
Examples:
{"tool_name": "get_passenger_details", "args": {"seat_id": "1A"}}
{"tool_name": "assign_seat", "args": {"passenger_id": "PAX-003", "target_seat_id": "3A"}}
{"tool_name": "swap_seats", "args": {"passenger_id_1": "PAX-003", "passenger_id_2": "PAX-007"}}

STRATEGY:
1. Fetch passenger details to discover preferences, prioritising passengers with the most constraints (legroom+window first, then window-only, then unconstrained).
2. Assign passengers in order of constraint strictness — most constrained first:
   a. Passengers with both paid_legroom and paid_window: target legroom window seats.
   b. Passengers with paid_legroom only: target remaining legroom seats.
   c. Passengers with paid_window only: target remaining window seats.
   d. All other passengers: fill any open seat in the correct cabin class.
3. If legroom seats exist on AC-2, reserve them for passengers who paid for legroom — do not waste them on unconstrained passengers.
4. Use swap_seats to fix any misplacements discovered after assignment.
5. You can read the AC-2 layout from the observation to identify which seats are windows or have legroom.

IMPORTANT:
- NEVER assign a passenger to the wrong cabin class. This is the highest priority and cannot be violated.
- Always fetch passenger details before assigning passengers whose preferences you do not yet know.
- Each tool call is one step. You can see the current state of both aircraft after each step.
- Use the step limit efficiently — avoid redundant fetches for passengers you have already queried."""

# ---------------------------------------------------------------------------
# Task definitions: (task_name, task_id, max_steps)
# ---------------------------------------------------------------------------
TASKS = [
    ("task_easy",   "easy",   24),
    ("task_medium", "medium", 60),
    ("task_hard",   "hard",   60),
]

# ---------------------------------------------------------------------------
# State and Prompt formatting functions
# ---------------------------------------------------------------------------
def format_main_task(task_id: str) -> str:
    if task_id == "easy":
        return "Task: Reassign all 8 passengers from Aircraft-1 (AC-1) to Aircraft-2 (AC-2), respecting cabin class."
    if task_id == "hard":
        return "Task: Reassign all 20 passengers from Aircraft-1 (AC-1) to Aircraft-2 (AC-2) respecting cabin class, window seat preferences, and extra legroom preferences."
    return "Task: Reassign all 20 passengers from Aircraft-1 (AC-1) to Aircraft-2 (AC-2)."

def format_state(obs) -> str:
    parts = []

    parts.append(
        f"=== Step {obs.step_count}/{obs.max_steps} | "
        f"Passengers remaining: {obs.passengers_remaining}/{obs.passengers_total} ==="
    )

    parts.append(f"\nAC-1 layout: {json.dumps(obs.ac1_layout['layout'], indent=2)}")
    parts.append(f"AC-2 layout: {json.dumps(obs.ac2_layout['layout'], indent=2)}")

    parts.append(f"\nAC-1 seats still occupied (passengers to reassign): {obs.ac1_seats_occupied}")

    if obs.ac2_seat_assignments:
        parts.append(f"\nAC-2 current assignments:")
        for seat_id, pax_id in sorted(obs.ac2_seat_assignments.items()):
            parts.append(f"  {seat_id} -> {pax_id}")
    else:
        parts.append(f"\nAC-2: No passengers assigned yet.")

    return "\n".join(parts)

def format_instruction() -> str:
    return "Choose your next tool call. Respond with ONLY a JSON object."

def format_result(item: dict) -> str:
    parts = []
    if item.get("result") is not None:
        parts.append(f"Last tool result: {json.dumps(item['result'], indent=2)}")
    if item.get("reward") is not None:
        parts.append(f"Reward: {item['reward']:.2f} ({item.get('reward_reason', 'unknown')})")

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
        # First query — include full task description + state
        user_content = "\n\n".join([
            format_main_task(task_id),
            format_state(obs),
            format_instruction()
        ])
        messages.append({"role": "user", "content": user_content})
    else:
        # Subsequent queries — replay recent history as interleaved turns
        messages.append({"role": "user", "content": format_main_task(task_id)})

        recent_history = conversation_history[-6:]
        for i, item in enumerate(recent_history):
            messages.append({"role": "assistant", "content": json.dumps(item["action"])})

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
        parsed = parse_llm_response(response_text)

        if parsed is None:
            return None

        return parsed
    except Exception as exc:
        return None


# ---------------------------------------------------------------------------
# Fallback action
# ---------------------------------------------------------------------------
def fallback_action(obs) -> dict:
    """Simple fallback: fetch the first unqueried AC-1 seat."""
    if obs.ac1_seats_occupied and obs.ac2_seats_available:
        seat_id = obs.ac1_seats_occupied[0]
        return {"tool_name": "get_passenger_details", "args": {"seat_id": seat_id}}
    return {"tool_name": "get_passenger_details", "args": {"seat_id": "1A"}}


# ---------------------------------------------------------------------------
# Mandatory logging
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Per-task episode runner
# ---------------------------------------------------------------------------
async def run_task(
    task_name: str,
    task_id: str,
    max_steps: int,
    client: OpenAI,
) -> None:
    env = SeatReassignmentEnv(base_url="http://localhost:8000")

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    conversation_history: list = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action_dict = get_agent_action(client, obs, conversation_history, task_id)

            if action_dict is None:
                action_dict = fallback_action(obs)

            action_summary = f"{action_dict['tool_name']}({json.dumps(action_dict['args'])})"


            result = await env.step(SeatReassignmentAction(
                tool_name=action_dict["tool_name"],
                args=action_dict["args"],
            ))
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
            log_step(step=step, action=action_summary, reward=reward, done=done, error=error)

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

    except Exception as e:
        pass

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main — loop over all tasks
# ---------------------------------------------------------------------------
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_name, task_id, max_steps in TASKS:
        await run_task(task_name, task_id, max_steps, client)


if __name__ == "__main__":
    asyncio.run(main())
